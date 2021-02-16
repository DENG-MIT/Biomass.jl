using OrdinaryDiffEq, Flux, Optim, Random, Plots
using Zygote
using ForwardDiff
using DiffEqSensitivity
using LinearAlgebra
using Statistics
using ProgressBars, Printf
using Flux.Optimise: update!
using Flux.Losses: mae, mse
using BSON: @save, @load
using DelimitedFiles

is_restart = false
n_epoch = 4000;
n_plot = 20;
# opt = ADAMW(0.005, (0.9, 0.999), 1.f-5);
opt = Flux.Optimiser(ExpDecay(5e-3, 0.2, 1000 * 10, 1e-5), 
                     ADAMW(0.005, (0.9, 0.999), 1.f-6));
grad_max = 1.f1;
maxiters = 10000;

ns = 5;
nr = 5;

lb = 1.e-6;
llb = 1e-6;
rb = 1e-3;
ub = 1.e1;

u0 = zeros(ns);
u0[1] = 1.0

l_exp = 1:14
n_exp = length(l_exp)
i_exp = 1;

l_train = []
l_val = []
for i in 1:n_exp
    j = l_exp[i]
    if !(j in [2, 6, 9, 12])
        push!(l_train, i)
    else
        push!(l_val, i)
    end
end

# ode_solver = AutoTsit5(KenCarp4(autodiff=false));
ode_solver = AutoTsit5(Rosenbrock23(autodiff=false));
# ode_solver = KenCarp4(autodiff=false);

function load_exp(filename)
    exp_data = readdlm(filename)  # [t, T, m]
    index = indexin(unique(exp_data[:, 1]), exp_data[:, 1])
    exp_data = exp_data[index, :]
    exp_data[:, 3] = exp_data[:, 3] / maximum(exp_data[:, 3])
    return exp_data
end

l_exp_data = [];
l_exp_info = zeros(Float32, length(l_exp), 3);
for (i_exp, value) in enumerate(l_exp)
    filename = string("exp_data/expdata_no", string(value), ".txt");
    exp_data = Float32.(load_exp(filename));

    if value == 4
        exp_data = exp_data[1:60, :]
    elseif value == 5
        exp_data = exp_data[1:58, :]
    elseif value == 6
        exp_data = exp_data[1:60, :]
    elseif value == 7
        exp_data = exp_data[1:71, :]
    end

    push!(l_exp_data, exp_data);
    T0 = exp_data[1, 2];  # initial temperature, K
    l_exp_info[i_exp, 1] = T0;
end
l_exp_info[:, 2] = readdlm("exp_data/beta.txt")[l_exp];
l_exp_info[:, 3] = log.(readdlm("exp_data/ocen.txt")[l_exp] .+ llb);

np = nr * (ns * 2 + 3) + 1
p = randn(Float32, np) .* 1.f-2;
p[1:nr] .+= 0.8;
p[nr * (ns * 2 + 1) + 1:nr * (ns * 2 + 2)] .+= 0.8;
p[nr * (ns * 2 + 2) + 1:nr * (ns * 2 + 3)] .+= 0.1;
p[nr * (ns * 2 + 3) + 1] = 0.1;


function p2vec(p)
    slope = p[nr * (ns + 3) + 1] .* 1.f2;
    w_b = p[1:nr] .* slope;
    w_b = clamp.(w_b, 0, 50);

    w_in = reshape(p[nr + 1:nr * (ns + 1)], ns, nr);
    w_out = reshape(p[nr * (ns + 1) + 1:nr * (ns * 2 + 1)], ns, nr);

    # i = 1
    # for j in 1:ns - 2
    #     w_out[j, i] = -1.0
    #     i = i + 1
    # end
    @. w_out[1, :] = clamp(w_out[1, :], -3, 0)
    @. w_out[end, :] = clamp(abs(w_out[end, :]), 0, 3)

    w_out[ns - 1:ns - 1, :] .= -sum(w_out[1:ns - 2, :], dims=1) .- sum(w_out[ns:ns, :], dims=1)

    w_in_Ea = abs.(p[nr * (ns * 2 + 1) + 1:nr * (ns * 2 + 2)] .* slope .* 10.f0);
    w_in_Ea = clamp.(w_in_Ea, 0.f0, 300.f0);

    w_in_ocen = abs.(p[nr * (ns * 2 + 2) + 1:nr * (ns * 2 + 3)]);
    w_in_ocen = clamp.(w_in_ocen, 0.f0, 1.5);
    # w_in_ocen[1:end] .= 0;

    w_in = clamp.(-w_out, 0.f0, 4.f0) .* abs.(w_in);
    w_in = vcat(w_in, w_in_Ea', w_in_ocen');
    
    return w_in, w_b, w_out
end

function display_p(p)
    w_in, w_b, w_out = p2vec(p);
    println("species (column) reaction (row)")
    println("w_in")
    show(stdout, "text/plain", round.(w_in', digits=3))

    println("\nw_b")
    show(stdout, "text/plain", round.(w_b', digits=3))

    println("\nw_out")
    show(stdout, "text/plain", round.(w_out', digits=3))
    println("\n\n")
end
# display_p(p);

R = -1.f0 / 8.314f-3  # universal gas constant, kJ/mol*K

function getsampletemp(t, T0, beta)
    if beta < 100
        T = T0 + beta / 60 * t  # K/min to K/s
    else
        tc = [999., 1059.] .* 60.;
        Tc = [beta, 370., 500.] .+ 273.;
        HR = 40.0 / 60.0;
        if t <= tc[1]
            T = Tc[1]
        elseif t <= tc[2]
            T = minimum([Tc[1] + HR * (t - tc[1]), Tc[2]]);
        else
            T = minimum([Tc[2] + HR * (t - tc[2]), Tc[3]]);
        end
    end
    return T
end

function crnn!(du, u, p, t)
    logX = @. log(clamp(u, llb, ub));
    T = getsampletemp(t, T0, beta)
    w_in_x = w_in' * vcat(logX, R / T, ocen);
    du .= w_out * (@. exp(w_in_x + w_b));
end

function makeprob(i_exp, p)
    exp_data = l_exp_data[i_exp];
    tlist = @views exp_data[:, 1];
    tspan = (tlist[1], tlist[end]);
    prob = ODEProblem(crnn!, u0, tspan, p, reltol=rb, abstol=lb);
    return prob, tlist
end

sense = BacksolveAdjoint(checkpointing=true; autojacvec=ZygoteVJP());
function cost_singleexp(prob, exp_data)
    sol = solve(prob, alg=ode_solver, saveat=@views(exp_data[:, 1]), 
                sensalg=sense, verbose=false, maxiters=maxiters);
    pred = Array(sol)
    masslist = sum(clamp.(@views(pred[1:end - 1, :]), 0, Inf), dims=1)';
    gaslist = clamp.(@views(pred[end, :]), 0, Inf);

    loss = mae(masslist, @views(exp_data[1:length(masslist), 3]))
    if ocen < 1000
        loss += mae(gaslist, 1 .- @views(exp_data[1:length(masslist), 3]))
    end

    if sol.retcode == :Success
        nothing
    else
        println("ode solver failed")
    end
    return loss
end

function loss_neuralode(p, i_exp)
    global T0, beta, ocen = l_exp_info[i_exp, :];
    global w_in, w_b, w_out = p2vec(p);
    prob, tlist = makeprob(i_exp, p);
    exp_data = l_exp_data[i_exp];
    loss = cost_singleexp(prob, exp_data);
    return loss
end
loss = loss_neuralode(p, 1)
# using BenchmarkTools
# @benchmark loss = loss_neuralode(p, 1)
# @benchmark grad = ForwardDiff.gradient(x -> loss_neuralode(x, 1), p)

function plot_sol(sol, exp_data, Tlist, cap, sol0=nothing)

    ts = sol.t / 60
    plt = plot(exp_data[:, 1] / 60, exp_data[:, 3], 
               seriestype=:scatter, label="Exp");
    
    plot!(plt, ts, sum(clamp.(sol[1:end - 1, :], 0, ub), dims=1)', lw=2, 
          legend=:left, label="Model");

    if sol0 !== nothing
        plot!(plt, sol0.t / 60, sum(sol0[1:end - 1, :], dims=1)', 
              label="initial model");
    end
    xlabel!(plt, "Time [min]");
    ylabel!(plt, "Mass");
    title!(plt, cap);

    plt2 = twinx();
    plot!(plt2, exp_data[:, 1] / 60, Tlist, lw=2, ls=:dash, 
          legend=:right, label="T");
    ylabel!(plt2, "Temp");

    p2 = plot(ts, sol[1, :], legend=:right, label="Cellulose")
    for i in 2:ns
        plot!(p2, ts, sol[i, :], label="S$i")
    end
    xlabel!(p2, "Time [min]");
    ylabel!(p2, "Mass");

    plt = plot(plt, p2, layout=@layout [a ; b])
    plot!(plt, size=(800, 800))
    return plt
end

cbi = function (p, i_exp)
    global T0, beta, ocen = l_exp_info[i_exp, :];
    global w_in, w_b, w_out = p2vec(p);
    prob, tlist = makeprob(i_exp, p);
    sol = solve(prob, alg=ode_solver, reltol=1e-3, abstol=1e-6, saveat=tlist);
    Tlist = copy(sol.t)
    for (i, t) in enumerate(sol.t)
        Tlist[i] = getsampletemp(t, T0, beta)
    end
    value = l_exp[i_exp]
    plt = plot_sol(sol, l_exp_data[i_exp], Tlist, "exp_$value");
    png(plt, string("figs/pred_exp_", value))
    return false
end

l_loss_train = []
l_loss_val = []
list_grad = []
iter = 1
cb = function (p, loss_train, loss_val, g_norm)
    global l_loss_train, l_loss_val, list_grad, iter
    push!(l_loss_train, loss_train)
    push!(l_loss_val, loss_val)
    push!(list_grad, g_norm)

    if iter % n_plot == 0
        display_p(p)
        list_exp = randperm(n_exp)[1]
        println("min loss ", minimum(l_loss_train))
        println("update plot ", l_exp[list_exp])
        for i_exp in list_exp
            cbi(p, i_exp)
        end

        plt_loss = plot(l_loss_train, xscale=:identity, yscale=:log10, label="train")
        plot!(plt_loss, l_loss_val, xscale=:identity, yscale=:log10, label="val")
        plt_grad = plot(list_grad, xscale=:identity, yscale=:log10, label="grad_norm")
        xlabel!(plt_loss, "Epoch")
        xlabel!(plt_grad, "Epoch")
        ylabel!(plt_loss, "Loss")
        ylabel!(plt_grad, "Grad Norm")
        ylims!(plt_loss, (-Inf, 1e0))
        plt_all = plot([plt_loss, plt_grad]..., legend=:top)
        png(plt_all, "figs/loss_grad")

        @save "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val list_grad iter
    end
    iter += 1
end

if is_restart
    @load "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val list_grad iter
    iter += 1
    # opt = ADAMW(0.005, (0.9, 0.999), 1.f-5);
end

epochs = ProgressBar(iter:n_epoch);
loss_epoch = zeros(Float32, n_exp);
grad_norm = zeros(Float32, n_exp);

for epoch in epochs
    global p
    for i_exp in randperm(n_exp)
        if i_exp in l_val
            continue
        end
        grad = ForwardDiff.gradient(x -> loss_neuralode(x, i_exp), p);
        grad_norm[i_exp] = norm(grad, 2)
        if grad_norm[i_exp] > grad_max
            grad = grad ./ grad_norm[i_exp] .* grad_max
        end
        update!(opt, p, grad)
    end
    for i_exp in 1:n_exp
        loss_epoch[i_exp] = loss_neuralode(p, i_exp)
    end
    loss_train = mean(loss_epoch[l_train])
    loss_val = mean(loss_epoch[l_val])
    grad_mean = mean(grad_norm[l_train])
    set_description(epochs, 
                    string(@sprintf("Loss train: %.4e val: %.4e grad: %.2e", 
                            loss_train, loss_val, grad_mean)))
    cb(p, loss_train, loss_val, grad_mean)
end

@sprintf("Min Loss train: %.4e val: %.4e", minimum(l_loss_train), minimum(l_loss_val))

for i_exp in randperm(n_exp)
    cbi(p, i_exp)
end