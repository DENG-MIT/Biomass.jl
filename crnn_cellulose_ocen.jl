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
n_epoch = 100000;
n_plot = 10;
opt = ADAMW(0.005, (0.9, 0.999), 1.f-5);
grad_max = 1.f2;

ns = 6;
nr = 8;

u0 = zeros(ns);
u0[1] = 1.0

lb = 1.e-4;
llb = 1e-12;
ub = 1.e1;

# l_exp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
# l_exp = [8, 9, 10, 11, 12, 13];
l_exp = 1:14
n_exp = length(l_exp)
i_exp = 1;

ode_solver = AutoTsit5(TRBDF2(autodiff=false));

function load_exp(filename)
    exp_data = readdlm(filename)  # [t, T, m]
    index = indexin(unique(exp_data[:, 1]), exp_data[:, 1])
    exp_data = exp_data[index, :]
    exp_data[:, 3] = exp_data[:, 3] / maximum(exp_data[:, 3])
    return exp_data
end

l_exp_data = [];
l_exp_info = zeros(Float64, length(l_exp), 3);
for (i_exp, value) in enumerate(l_exp)
    filename = string("exp_data/expdata_no", string(value), ".txt");
    exp_data = load_exp(filename);

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

np = nr * (ns + 3) + 1
p = randn(Float64, np) .* 1.f-2;
p[1:nr] .+= 0.8;
p[nr * (ns + 1) + 1:nr * (ns + 2)] .+= 0.8;
p[nr * (ns + 2) + 1:nr * (ns + 3)] .+= 0.1;
p[nr * (ns + 3) + 1] = 0.1;


function p2vec(p)
    slope = p[nr * (ns + 3) + 1] .* 1.f2;
    w_b = p[1:nr] .* slope;
    w_b = clamp.(w_b, 0, 50);

    w_out = reshape(p[nr + 1:nr * (ns + 1)], ns, nr);

    @. w_out[1, :] = clamp(w_out[1, :], -3, 0)
    @. w_out[end, :] = clamp(abs(w_out[end, :]), 0, 3)

    i = 1
    for j in 1:ns - 2
        w_out[j, i] = -1.0
        i = i + 1
    end
    w_out[ns - 1:ns - 1, :] .= -sum(w_out[1:ns - 2, :], dims=1) .- sum(w_out[ns:ns, :], dims=1)

    w_in_Ea = abs.(p[nr * (ns + 1) + 1:nr * (ns + 2)] .* slope .* 10.f0);
    w_in_Ea = clamp.(w_in_Ea, 0.f0, 300.f0);

    w_in_ocen = abs.(p[nr * (ns + 2) + 1:nr * (ns + 3)]);
    w_in_ocen = clamp.(w_in_ocen, 0.f0, 1.5);
    w_in_ocen[1:ns] .= 0;

    w_in = clamp.(-w_out, 0.f0, 4.f0);
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
display_p(p)

R = -1.f0 / 8.314f-3  # universal gas constant, kJ/mol*K
function crnn!(du, u, p, t)
    logX = @. log(clamp(u, lb, ub));
    T = getsampletemp(t, T0, beta)
    w_in_x = w_in' * vcat(logX, R / T, ocen);
    du .= w_out * (@. exp(w_in_x + w_b));
end

function makeprob(i_exp, p)
    exp_data = l_exp_data[i_exp];
    tlist = exp_data[:, 1];
    tspan = (tlist[1], tlist[end]);
    prob = ODEProblem(crnn!, u0, tspan, p, reltol=1e-2, abstol=lb);
    return prob, tlist
end

# sense = BacksolveAdjoint(checkpointing=true);
sense = ForwardDiffSensitivity(convert_tspan=false)
function cost_singleexp(prob, exp_data)
    sol = solve(prob, alg=ode_solver, saveat=exp_data[:, 1], 
                sensalg=sense, verbose=false, maxiters=10000);
    pred = Array(sol)

    masslist = sum(clamp.(pred[1:end - 1, :], -ub, ub), dims=1)';
    gaslist = clamp.(pred[end, :], -ub, ub);

    loss = mae(masslist, exp_data[1:length(masslist), 3])
    if ocen < 1000
        loss += mae(gaslist, 1 .- exp_data[1:length(masslist), 3])
    end

    if sol.retcode == :Success
        nothing
    else
        @show "ode solver failed, set losss = $loss"
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

function plot_sol(sol, exp_data, Tlist, cap, sol0=nothing)

    ts = sol.t / 60
    plt = plot(exp_data[:, 1] / 60, exp_data[:, 3], 
               seriestype=:scatter, label="Exp");
    
    plot!(plt, ts, sum(sol[1:end - 1, :], dims=1)', lw=2, 
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

list_loss = []
list_grad = []
iter = 1
cb = function (p, loss_mean, g_norm)
    global list_loss, list_grad, iter
    push!(list_loss, loss_mean)
    push!(list_grad, g_norm)

    if iter % n_plot == 0
        display_p(p)
        list_exp = randperm(n_exp)[1]
        println("min loss ", minimum(list_loss))
        println("update plot ", l_exp[list_exp])
        for i_exp in list_exp
            cbi(p, i_exp)
        end

        plt_loss = plot(list_loss, xscale=:identity, yscale=:log10, label="loss")
        plt_grad = plot(list_grad, xscale=:identity, yscale=:log10, label="grad_norm")
        xlabel!(plt_loss, "Epoch")
        xlabel!(plt_grad, "Epoch")
        ylabel!(plt_loss, "Loss")
        ylabel!(plt_grad, "Grad Norm")
        plt_all = plot([plt_loss, plt_grad]..., legend=:right)
        png(plt_all, "figs/loss_grad")

        @save "./checkpoint/mymodel.bson" p opt list_loss list_grad iter
    end
    iter += 1
end

if is_restart
    @load "./checkpoint/mymodel.bson" p opt list_loss list_grad iter
    iter += 1
    # opt = ADAMW(0.001, (0.9, 0.999), 1.f-6);
end

epochs = ProgressBar(iter:n_epoch);
loss_epoch = zeros(Float64, n_exp);
grad_norm = zeros(Float64, n_exp);

for epoch in epochs
    global p
    for i_exp in randperm(n_exp)
        grad = ForwardDiff.gradient(x -> loss_neuralode(x, i_exp), p);
        grad_norm[i_exp] = norm(grad, 2)
        if grad_norm[i_exp] > grad_max
            grad = grad ./ grad_norm[i_exp] .* grad_max
        end
        update!(opt, p, grad)
    end
    for i_exp in randperm(n_exp)
        loss_epoch[i_exp] = loss_neuralode(p, i_exp)
    end
    loss_mean = mean(loss_epoch)
    grad_mean = mean(grad_norm)
    set_description(epochs, string(@sprintf("Loss: %.4e grad: %.2e", loss_mean, grad_mean)))
    cb(p, loss_mean, grad_mean)
end

for i_exp in randperm(n_exp)
    cbi(p, i_exp)
end

# # BFGS
# function f(p)
#     loss = 0
#     for i_exp in 1:n_exp
#         loss += loss_neuralode(p, i_exp)
#     end
#     return loss / n_exp
# end

# function g!(G, p)
#     G .= ForwardDiff.gradient(x -> f(x), p);
# end

# G = zeros(size(p));
# loss = f(p)
# g!(G, p);
# grad_norm = norm(G, 2)
# println("bfgs initial loss $loss grad $grad_norm")

# pp = p;
# for ii in 1:10
#     res = optimize(f, g!, pp,
#                     BFGS(),
#                     Optim.Options(g_tol=1e-12, iterations=50,
#                                   store_trace=true, show_trace=true))
#     pp = res.minimizer
#     loss = f(pp)
#     g!(G, pp)
#     grad_norm = norm(G, 2)
#     println("bfgs iter $ii loss $loss grad $grad_norm")
#     for i_exp in randperm(n_exp)
#         cbi(pp, i_exp)
#     end
# end
