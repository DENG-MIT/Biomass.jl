using DifferentialEquations, Optim, Plots, DelimitedFiles
using Statistics:mean
# using DiffEqParamEstim
# using DiffEqSensitivity
# using Sundials

function richter2019!(du, u, p, t)

    R = 8.314E-3    # universal gas constant, kJ/mol*K

    # [Ea1-3, logA1-3, v3, T0, beta]
    Ea = p[1:3]   # kJ/mol
    logA = p[4:6]
    v = p[7]
    
    T0 = p[8]
    beta = p[9]

    u = clamp.(u, 1e-30, 1e2)
    cell = u[1]
    cella = u[2]
    char = u[3]

    if beta < 100
        T = T0 + beta / 60 * t  # K/min to K/s
    else
        T = beta + 273.15  # iso-thermal
    end

    k = (10 .^ logA) .* exp.(- Ea ./ R ./ T)

    w1 = k[1] * (cell)
    w2 = k[2] * (cella)
    w3 = k[3] * (cella)

    du[1] = - w1
    du[2] = w1 - (w2 + w3)
    du[3] = w3 * v
end

function plot_sol(sol, exp_data, cap, sol0=nothing)
    
    plt = plot(sol.t/60, sum(sol, dims=1)', label="model");
    plot!(plt, exp_data[:, 1]/60, exp_data[:, 3], label="exp");
    if sol0 !== nothing
        plot!(plt, sol0.t/60, sum(sol0, dims=1)', label="initial model");
    end
    xlabel!(plt, "time [min]");
    ylabel!(plt, "mass");
    title!(plt, cap);
    
    return plt
end

function load_exp(filename)
    exp_data = readdlm(filename)
    # [t, T, m]
    index = indexin(unique(exp_data[:, 1]), exp_data[:, 1])
    exp_data = exp_data[index, :]
    exp_data[:, 3] = exp_data[:, 3] / maximum(exp_data[:, 3])
    return exp_data
end

ode_solver = Kvaerno5();
# Rodas5, Rodas4P, Kvaerno5, KenCarp4, https://diffeq.sciml.ai/v4.0/solvers/ode_solve.html

l_exp_data = [];
l_exp = [1,2,3,8,9,10];
l_exp_info = zeros(length(l_exp), 2);
for (i_exp, value) in enumerate(l_exp)
    filename = string("exp_data/expdata_no" , string(value) , ".txt");
    exp_data = load_exp(filename);
    push!(l_exp_data, exp_data);
    T0 = exp_data[1, 2];  # initial temperature, K
    l_exp_info[i_exp, 1] = T0;
end
l_exp_info[:, 2] = readdlm("exp_data/beta.txt")[l_exp];

# [Ea1-3, logA1-3, v3, T0, beta]
# Ea = [243, 198, 153]  # kJ/mol   # 150
# logA = [19.5, 14.1, 9.69]  # 15
# v = 0.35  #R3

Ea = ones(3) * 150;
logA = ones(3) * 15;
v = 0.5;

u0 = [1.0, 0.0, 0.0];

function makeprob(i_exp)
    exp_data = l_exp_data[i_exp];
    tlist = exp_data[:, 1];
    T0, beta = l_exp_info[i_exp, :];

    tspan = (tlist[1], tlist[end]);
    p = vcat(Ea, logA, v, T0, beta);
    prob = ODEProblem(richter2019!, u0, tspan, p);

    return prob, tlist
end

for (i_exp, value) in enumerate(l_exp)
    prob, tlist = makeprob(i_exp);

    sol = solve(prob, alg=ode_solver,
        reltol=1e-3, abstol=1e-6,
        saveat=tlist);

    plt = plot_sol(sol, l_exp_data[i_exp], "initial_p_$value");

    savefig(plt, "fig/initial_p_$value");
end

# Training
function cost_singleexp(prob, exp_data)
    sol = solve(prob, alg=ode_solver,
                reltol=1e-3, abstol=1e-6, saveat=exp_data[:, 1],
                maxiters=10000, verbose=false);
    masslist = sum(sol, dims=1)';

    if sol.retcode == :Success
        loss = mean((masslist - exp_data[:, 3]).^2);
    else
        loss = 1e3
        @show "ode solver failed, set losss = $loss"
    end
    return loss
end

function cost_function(p)
    l_loss = zeros(length(l_exp));
    # TODO: not sure if we can makeprob outside
    prob, tlist = makeprob(1);
    for (i_exp, value) in enumerate(l_exp)
        p_ = vcat(p, l_exp_info[i_exp, :]);
        exp_data = l_exp_data[i_exp];
        tlist = exp_data[:, 1];
        tspan = (tlist[1], tlist[end]);
        # prob = ODEProblem(richter2019!, u0, tspan, p_);
        prob = remake(prob; tspan=tspan, p = p_);
        l_loss[i_exp] = cost_singleexp(prob, exp_data);
    end
    # @show mean(l_loss)
    return mean(l_loss)
end

lb_Ea = ones(3) * 10.0;
ub_Ea = ones(3) * 300.0;
lb_logA = ones(3) * 0;
ub_logA = ones(3) * 23;
lb_v = ones(1) * 0;
ub_v = ones(1);
lb = vcat(lb_Ea, lb_logA, lb_v);
ub = vcat(ub_Ea, ub_logA, ub_v);
p0 = vcat(Ea, logA, v);

cost_function(p0)

# global optimization
# res = optimize(cost_function, lb, ub, p0, 
#                NelderMead(),
#                Optim.Options(g_tol = 1e-8,
#                              iterations = 10000,
#                              store_trace = false,
#                              show_trace = false,
#                              show_every=1))

# using Evolutionary
# res = Evolutionary.optimize(cost_function, p0,
#              GA(populationSize = 100, selection = susinv,
#                 crossover = discrete, mutation = domainrange(ones(3))))

using CMAEvolutionStrategy
# https://github.com/jbrea/CMAEvolutionStrategy.jl

# TODO: multi_threading

res = minimize(cost_function, p0, 0.2,
                lower = lb, upper = ub,
                verbosity = 1,
                ftol = 1e-9,
                popsize = 50,
                parallel_evaluation = false,
                multi_threading = false,
                )

# for ii in 1:5
#     res = minimize(cost_function, xbest(res), 0.05,
#                 lower = lb, upper = ub,
#                 verbosity = 1,
#                 ftol = 1e-9,
#                 popsize = 50,
#                 parallel_evaluation = false,
#                 multi_threading = false,
#                 )
# end

# local optimization
# res = optimize(cost_function, lb, ub, xbest(res), 
#                Fminbox(LBFGS()),
#                Optim.Options(g_tol = 1e-8, 
#                              f_tol= 1e-10,
#                              iterations = 100,
#                              store_trace = false,
#                              show_trace = true,
#                              show_every=1))

# p = res.minimizer

prob, tlist = makeprob(1);
for (i_exp, value) in enumerate(l_exp)
    p_ = vcat(p, l_exp_info[i_exp, :])
    exp_data = l_exp_data[i_exp]
    tlist = exp_data[:, 1]
    tspan = (tlist[1], tlist[end])
    prob = remake(prob; tspan=tspan, p=p_)
    sol = solve(prob, alg=ode_solver,
        reltol=1e-3, abstol=1e-6,
        saveat=tlist);

    p0_ = vcat(Ea, logA, v, l_exp_info[i_exp, :]);
    prob = remake(prob; tspan=tspan, p=p0_)
    sol0 = solve(prob, alg=ode_solver,
                 reltol=1e-3, abstol=1e-6,
                 saveat=tlist);

    plt = plot_sol(sol, exp_data, "optimized_p_$value", sol0)
    savefig(plt, "fig/optimized_p_$value")
end