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

is_restart = true
n_epoch = 100000;
n_plot = 10;
opt = Flux.Optimiser(ExpDecay(5e-3, 0.2, 300 * 10, 1e-6), ADAMW(0.005, (0.9, 0.999), 1.f-8));
grad_max = 1.f2;
maxiters = 5000;

p_cutoff = 0.1

ns = 5;
nr = 6;

lb = 1.e-6;
llb = 1e-6;
rb = 1e-3;
ub = 1.e2;

u0 = zeros(ns);
u0[1] = 1.0

l_exp = 1:14
n_exp = length(l_exp)
i_exp = 1;

# ode_solver = AutoTsit5(KenCarp4(autodiff=false));
ode_solver = AutoTsit5(Rosenbrock23(autodiff=false));

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

    # if value == 4
    #     exp_data = exp_data[1:60, :]
    # elseif value == 5
    #     exp_data = exp_data[1:58, :]
    # elseif value == 6
    #     exp_data = exp_data[1:60, :]
    # elseif value == 7
    #     exp_data = exp_data[1:71, :]
    # end

    push!(l_exp_data, exp_data);
    T0 = exp_data[1, 2];  # initial temperature, K
    l_exp_info[i_exp, 1] = T0;
end
l_exp_info[:, 2] = readdlm("exp_data/beta.txt")[l_exp];
l_exp_info[:, 3] = log.(readdlm("exp_data/ocen.txt")[l_exp] .+ llb);

np = nr * (ns + 4) + 1
p = randn(Float64, np) .* 1.f-2;
p[1:nr] .+= 0.8;  # w_b
p[nr * (ns + 1) + 1:nr * (ns + 2)] .+= 0.8;  # w_out
p[nr * (ns + 2) + 1:nr * (ns + 4)] .+= 0.1;  # w_Ea
p[end] = 0.1;  # slope

function p2vec(p)
    slope = p[end] .* 1.f2;
    w_b = p[1:nr] .* slope;
    w_b = clamp.(w_b, 0, 50);
    w_out = reshape(p[nr + 1:nr * (ns + 1)], ns, nr);

    i = 1
    for j in 1:ns - 2
        w_out[j, i] = -1.0
        i = i + 1
    end
    @. w_out[1, :] = clamp(w_out[1, :], -3, 0)
    @. w_out[end, :] = clamp(abs(w_out[end, :]), 0, 3)

    index = findall(abs.(w_out) .< p_cutoff);
    w_out[index] .= 0;

    w_out[ns - 1:ns - 1, :] .= -sum(w_out[1:ns - 2, :], dims=1) .- sum(w_out[ns:ns, :], dims=1)

    w_in_Ea = abs.(p[nr * (ns + 1) + 1:nr * (ns + 2)] .* slope .* 10.f0)
    w_in_Ea = clamp.(w_in_Ea, 0.f0, 300.f0)

    w_in_b = abs.(p[nr * (ns + 2) + 1:nr * (ns + 3)])

    w_in_ocen = abs.(p[nr * (ns + 3) + 1:nr * (ns + 4)])
    w_in_ocen = clamp.(w_in_ocen, 0.f0, 1.5)
    w_in_ocen[1:ns - 1] .= 0

    w_in = vcat(clamp.(-w_out, 0.f0, 4.f0), w_in_Ea', w_in_b', w_in_ocen')
    return w_in, w_b, w_out
end

function display_p(p)
    w_in, w_b, w_out = p2vec(p);
    println("\nspecies (column) reaction (row)")
    println("w_in | Ea | beta | n_ocen | lnA | w_out")
    show(stdout, "text/plain", round.(hcat(w_in', w_b, w_out'), digits=2))
    # println("\nw_out")
    # show(stdout, "text/plain", round.(w_out', digits=3))
    println("\n")
end
display_p(p)

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

R = -1.f0 / 8.314f-3  # universal gas constant, kJ/mol*K
function crnn!(du, u, p, t)
    logX = @. log(clamp(u, llb, ub));
    T = getsampletemp(t, T0, beta)
    w_in_x = w_in' * vcat(logX, R / T, log(T), ocen);
    du .= w_out * (@. exp(w_in_x + w_b));
end

function makeprob(i_exp, p)
    exp_data = l_exp_data[i_exp];
    tlist = @views exp_data[:, 1];
    tspan = (tlist[1], tlist[end]);
    prob = ODEProblem(crnn!, u0, tspan, p, reltol=rb, abstol=lb);
    return prob, tlist
end

sense = BacksolveAdjoint(checkpointing=true);
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

        plt_loss = plot(l_loss_train, xscale=:identity, yscale=:log10, label="train", legend=:best)
        plot!(plt_loss, l_loss_val, xscale=:identity, yscale=:log10, label="validation")
        plt_grad = plot(list_grad, xscale=:identity, yscale=:log10, legend=false)
        xlabel!(plt_loss, "Epoch")
        xlabel!(plt_grad, "Epoch")
        ylabel!(plt_loss, "Loss")
        ylabel!(plt_grad, "Gradient Norm")
        ylims!(plt_loss, (-Inf, 1e0))
        ylims!(plt_grad, (-Inf, 1e3))
        plt_all = plot([plt_loss, plt_grad]..., framestyle=:box, size=(800, 350))
        png(plt_all, "figs/loss_grad")

        @save "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val list_grad iter
    end
    iter += 1
end

if is_restart
    @load "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val list_grad iter
    iter += 1
    # opt = ADAMW(0.0001, (0.9, 0.999), 1.f-6);
end

display_p(p);
loss_epoch = zeros(Float64, n_exp);
for i_exp in 1:n_exp
    loss_epoch[i_exp] = loss_neuralode(p, i_exp)
end
loss_mean = mean(loss_epoch);
@sprintf("cutoff %.2f Loss: %.4e", p_cutoff, loss_mean)

w_in, w_b, w_out = p2vec(p);
pw = vcat(w_in, w_b', w_out)'
using DelimitedFiles
writedlm( "weights.csv",  pw, ',')


# Plot TGA data
varnames = ["Cellu", "S2", "S3", "S4", "S5", "Vola"]
for i_exp in [9, 11, 12]
    global T0, beta, ocen = l_exp_info[i_exp, :];
    global w_in, w_b, w_out = p2vec(p);
    prob, tlist = makeprob(i_exp, p);
    sol = solve(prob, alg=ode_solver, reltol=1e-3, abstol=1e-6, saveat=tlist);
    Tlist = copy(sol.t)
    for (i, t) in enumerate(sol.t)
        Tlist[i] = getsampletemp(t, T0, beta)
    end
    value = l_exp[i_exp]
    exp_data = l_exp_data[i_exp]
    plt = plot(Tlist, exp_data[:, 3], seriestype=:scatter, label="Exp", framestyle=:box);
    plot!(plt, Tlist, sum(sol[1:end - 1, :], dims=1)', lw=2, legend=:best, label="Model");
    xlabel!(plt, "Temperature [K]");
    ylabel!(plt, "Mass [-]");
    plot!(plt, xtickfontsize=11, ytickfontsize=11, xguidefontsize=12, yguidefontsize=12)
    plot!(plt, size=(400, 400), legend=false)
    # if i_exp == 11
    #     plot!(plt, size=(900, 300), legend=false)
    # else
    #     plot!(plt, size=(400, 400), legend=false)
    # end
    png(plt, string("figs/pred_exp_", value))

    list_plt = []
    for i in 1:ns
        plt = plot(Tlist, clamp.(sol[i, :], 0, Inf), 
                    framestyle=:box, xtickfontsize=11, ytickfontsize=11, xguidefontsize=12, yguidefontsize=12)
        xlabel!(plt, "Temperature [K]");
        ylabel!(plt, "[$(varnames[i])]");
        push!(list_plt, plt)
    end
    plt_all = plot(list_plt...)
    plot!(plt_all, size=(1200, 800), legend=false)
    png(plt_all, string("figs/pred_S_exp_", value))
end


# Plot TGA data
list_plt = []
for i_exp in 1:14
    global T0, beta, ocen = l_exp_info[i_exp, :];
    global w_in, w_b, w_out = p2vec(p);
    prob, tlist = makeprob(i_exp, p);
    sol = solve(prob, alg=ode_solver, reltol=1e-3, abstol=1e-6, saveat=tlist);
    Tlist = copy(sol.t)
    for (i, t) in enumerate(sol.t)
        Tlist[i] = getsampletemp(t, T0, beta)
    end
    exp_data = l_exp_data[i_exp]

    if beta < 100
        plt = plot(Tlist, exp_data[:, 3], seriestype=:scatter, label="Exp-$i_exp", framestyle=:box);
        plot!(plt, Tlist, sum(sol[1:end - 1, :], dims=1)', lw=2, legend=false);
        xlabel!(plt, "Temperature [K]");
        
    else
        plt = plot(sol.t / 60, exp_data[:, 3], seriestype=:scatter, label="Exp-$i_exp", framestyle=:box);
        plot!(plt, sol.t / 60, sum(sol[1:end - 1, :], dims=1)', lw=2, legend=false);
        xlabel!(plt, "Time [min]");
    end
    ylabel!(plt, "Mass [-] (No. $i_exp)");
    plot!(plt, xtickfontsize=11, ytickfontsize=11, xguidefontsize=12, yguidefontsize=12)
    
    push!(list_plt, plt)
end
plt_all = plot(list_plt..., layout=(7, 2))
plot!(plt_all, size=(800, 1200))
png(plt_all, string("figs/mass_summary"))