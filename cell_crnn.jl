using OrdinaryDiffEq, Flux, Optim, Random, Plots
using Zygote
using ForwardDiff
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
opt = ADAMW(0.001, (0.9, 0.999), 1.f-6);
# opt = RADAM(0.002, (0.9, 0.999))

ns = 6;
nr = 6;

u0 = zeros(ns);
u0[1] = 1.0

lb = 1.e-6;
ub = 1.e1;

l_exp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
n_exp = length(l_exp)
i_exp = 1;

# ode_solver = Rodas5(autodiff=false);
# ode_solver = Kvaerno5(autodiff=false);
ode_solver = Rosenbrock23(autodiff=false)
# Rodas5, Rodas4P, Kvaerno5, KenCarp4, 
# https://diffeq.sciml.ai/v4.0/solvers/ode_solve.html

function load_exp(filename)
    exp_data = readdlm(filename)  # [t, T, m]
    index = indexin(unique(exp_data[:, 1]), exp_data[:, 1])
    exp_data = exp_data[index, :]
    exp_data[:, 3] = exp_data[:, 3] / maximum(exp_data[:, 3])
    return exp_data
end

l_exp_data = [];
l_exp_info = zeros(Float64, length(l_exp), 2);
for (i_exp, value) in enumerate(l_exp)
    filename = string("exp_data/expdata_no", string(value), ".txt");
    exp_data = load_exp(filename);

    if i_exp == 4
        exp_data = exp_data[1:60, :]
    elseif i_exp == 5
        exp_data = exp_data[1:58, :]
    elseif i_exp == 6
        exp_data = exp_data[1:60, :]
    elseif i_exp == 7
        exp_data = exp_data[1:71, :]
    end

    push!(l_exp_data, exp_data);
    T0 = exp_data[1, 2];  # initial temperature, K
    l_exp_info[i_exp, 1] = T0;
end
l_exp_info[:, 2] = readdlm("exp_data/beta.txt")[l_exp];

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

# [Ea1-3, logA1-3, v3, T0, beta]
# Ea = [243, 198, 153]  # kJ/mol   # 150
# logA = [19.5, 14.1, 9.69]  # 15
# v = 0.35  # R3

p = randn(Float32, nr * (ns + 2) + 1) .* 1.f-2;
p[1:nr] .+= 0.8;
p[nr * (ns + 1) + 1:nr * (ns + 2)] .+= 0.8;
p[nr * (ns + 2) + 1] = 0.1;

function p2vec(p)
    slope = p[nr * (ns + 2) + 1] .* 1.f2;
    w_b = p[1:nr] .* slope;
    w_b = clamp.(w_b, -23, 50);

    w_out = reshape(p[nr + 1:nr * (ns + 1)], ns, nr);

    @. w_out[1, :] = clamp(w_out[1, :], -3, 0)
    @. w_out[end - 1:end, :] = clamp(abs(w_out[end - 1:end, :]), 0, 3)

    i = 1
    for j = 1:ns - 2
        w_out[j, i] = -1.0
        i = i + 1
    end

    w_out[ns - 2:ns - 2, :] .= -sum(w_out[1:ns - 3, :], dims=1) .- sum(w_out[ns - 1:ns, :], dims=1)

    w_in_Ea = abs.(p[nr * (ns + 1) + 1:nr * (ns + 2)] .* slope .* 10.f0);
    w_in_Ea = clamp.(w_in_Ea, 0.f0, 300.f0);

    w_in = clamp.(-w_out, 0.f0, 4.f0);
    w_in = vcat(w_in, w_in_Ea');
    
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

R = 8.314f-3    # universal gas constant, kJ/mol*K

function crnn!(du, u, p, t)
    logX = @. log(clamp(u, lb, ub));
    T = getsampletemp(t, T0, beta)
    w_in_x = w_in' * vcat(logX, -1.f0 / R / T);
    du .= w_out * (@. exp(w_in_x + w_b));
end

function richter2019!(du, u, p, t)
    u = clamp.(u, lb, ub)
    cell = u[1]
    cella = u[2]
    char = u[3]

    Ea, logA, b, v, slope = p2vec(p)

    # @show T0, beta

    T = getsampletemp(t, T0, beta)

    k = @. (10.f0^logA) * (T^b) * exp(- Ea / R / T)

    w1 = k[1] * (cell)
    w2 = k[2] * (cella)
    w3 = k[3] * (cella)

    du[1] = - w1
    du[2] = w1 - (w2 + w3)
    du[3] = w3 * v[1]

    # @show k, du
end

function makeprob(i_exp, p)
    exp_data = l_exp_data[i_exp];
    tlist = exp_data[:, 1];
    tspan = (tlist[1], tlist[end]);
    prob = ODEProblem(crnn!, u0, tspan, p, reltol=1e-2, abstol=lb);
    return prob, tlist
end

# Training
function cost_singleexp(prob, exp_data)
    sol = solve(prob, alg=ode_solver,
                saveat=exp_data[:, 1],
                maxiters=100000, verbose=false);
    
    pred = Array(sol)

    # masslist = clamp.(sum(sol, dims=1)', -ub, ub);
    masslist = sum(clamp.(pred[1:end - 1, :], -ub, ub), dims=1)';
    gaslist = clamp.(pred[end, :], -ub, ub);

    loss = mae(masslist, exp_data[1:length(masslist), 3])
    gasloss = mae(gaslist, 1 .- exp_data[1:length(masslist), 3])

    loss += gasloss

    if sol.retcode == :Success
        nothing
    else
        @show "ode solver failed, set losss = $loss"
    end
    return loss
end

function loss_neuralode(p, i_exp)
    global T0, beta = l_exp_info[i_exp, :];
    global w_in, w_b, w_out = p2vec(p);
    prob, tlist = makeprob(i_exp, p);
    exp_data = l_exp_data[i_exp];
    loss = cost_singleexp(prob, exp_data);
    return loss
end

loss = loss_neuralode(p, 1)

# using BenchmarkTools
# ode_solver = Rodas5(autodiff=false);
# Rodas5, Rodas4P, Kvaerno5, KenCarp4
# @benchmark loss = loss_neuralode(p, 1)

# Callback function to observe training
function plot_sol(sol, exp_data, Tlist, cap, sol0=nothing)
    
    plt = plot(exp_data[:, 1] / 60, exp_data[:, 3], seriestype=:scatter, label="exp");

    plot!(plt, sol.t / 60, sum(sol[1:end - 1, :], dims=1)', lw=2, label="model", legend=:bottomleft);
    
    if sol0 !== nothing
        plot!(plt, sol0.t / 60, sum(sol0[1:end - 1, :], dims=1)', label="initial model");
    end
    
    xlabel!(plt, "time [min]");
    ylabel!(plt, "mass");
    title!(plt, cap);

    plt2 = twinx();
    plot!(plt2, exp_data[:, 1] / 60, Tlist, lw=2, ls=:dash, label="T", legend=:left);
    ylabel!(plt2, "T");

    p2 = plot(sol.t / 60, sol[1, :], label="cell", legend=:bottomleft)
    plot!(p2, sol.t / 60, sol[2, :], label="cella")
    plot!(p2, sol.t / 60, sol[end, :], label="gas")
    xlabel!(p2, "time [min]");
    ylabel!(p2, "mass");

    plt = plot(plt, p2, layout=@layout [a ; b])
    return plt
end


for (i_exp, value) in enumerate(l_exp)

    @show i_exp, value

    global T0, beta = l_exp_info[i_exp, :];
    global w_in, w_b, w_out = p2vec(p);
    prob, tlist = makeprob(i_exp, p);

    sol = solve(prob, alg=ode_solver,
                reltol=1e-3, abstol=1e-6,
                saveat=tlist);

    Tlist = copy(sol.t)
    for (i, t) in enumerate(sol.t)
        Tlist[i] = getsampletemp(t, T0, beta)
    end

    plt = plot_sol(sol, l_exp_data[i_exp], Tlist, "exp_$value");

    savefig(plt, "figs/initial_p_$value");
end

cbi = function (p, i_exp)
    global T0, beta = l_exp_info[i_exp, :];
    global w_in, w_b, w_out = p2vec(p);
    prob, tlist = makeprob(i_exp, p);
    sol = solve(prob, alg=ode_solver,
                reltol=1e-3, abstol=1e-6,
                saveat=tlist);
    Tlist = copy(sol.t)
    for (i, t) in enumerate(sol.t)
        Tlist[i] = getsampletemp(t, T0, beta)
    end
    value = l_exp[i_exp]
    plt = plot_sol(sol, l_exp_data[i_exp], Tlist, "exp_$value");
    png(plt, string("figs/i_exp_", value))
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
        println("update plot for ", l_exp[list_exp])
        for i_exp in list_exp
            cbi(p, i_exp)
        end

        plt_loss = plot(list_loss, xscale=:log10, yscale=:log10, label="loss")
        png(plt_loss, "figs/loss")

        plt_grad = plot(list_grad, xscale=:log10, yscale=:log10, label="grad_norm")
        png(plt_grad, "figs/grad")

        @save "./checkpoint/mymodel.bson" p opt list_loss list_grad iter
    end
    iter += 1
end


if is_restart
    @load "./checkpoint/mymodel.bson" p opt list_loss list_grad iter
    iter += 1
end

epochs = ProgressBar(iter:n_epoch);
loss_epoch = zeros(Float64, n_exp);
grad_norm = zeros(Float64, n_exp);

opt = ADAMW(0.0001, (0.9, 0.999), 1.f-6);

for epoch in epochs
    global p
    for i_exp in randperm(n_exp)
        # Zygote forward mode AD
        grad = gradient(p) do x
            Zygote.forwarddiff(x) do x
                loss_neuralode(x, i_exp)
            end
        end
        grad = grad[1]

        grad_norm[i_exp] = norm(grad, 2)

        grad = grad ./ grad_norm[i_exp] .* 1.e2
        update!(opt, p, grad)
    end
    for i_exp in randperm(n_exp)
        loss_epoch[i_exp] = loss_neuralode(p, i_exp)
    end
    loss_mean = mean(loss_epoch)
    set_description(epochs, string(@sprintf("Loss: %.4e grad: %.2e", loss_mean, mean(grad_norm))))
    cb(p, loss_mean, mean(grad_norm))
end

for i_exp in randperm(n_exp)
    cbi(p, i_exp)
end