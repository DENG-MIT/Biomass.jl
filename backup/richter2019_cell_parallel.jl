# using Plots
# using DiffEqParamEstim
# using DiffEqSensitivity
# using Sundials

using Distributed
addprocs(12)

# Shared Process

@everywhere begin

    using DifferentialEquations, DelimitedFiles, Plots
    using Statistics:mean

    ode_solver = Kvaerno5();
    # Rodas5, Rodas4P, Kvaerno5, KenCarp4, 
    # https://diffeq.sciml.ai/v4.0/solvers/ode_solve.html
    l_exp = [1,2,3,4,5,6,7,8,9,10];

    function load_exp(filename)
        exp_data = readdlm(filename)  # [t, T, m]
        index = indexin(unique(exp_data[:, 1]), exp_data[:, 1])
        exp_data = exp_data[index, :]
        exp_data[:, 3] = exp_data[:, 3] / maximum(exp_data[:, 3])
        return exp_data
    end

    function getsampletemp(t, T0, beta, i_exp)

        if beta < 100
            T = T0 + beta / 60 * t  # K/min to K/s
        else
            if convert(Int16, i_exp) in [4, 5, 6, 7]  #Varhegyi 1994 iso-thermal
                tc=[999., 1059.] .* 60.;
                Tc=[beta, 370., 500.] .+ 273.;
                HR=40.0/60.0;
                if t <= tc[1]
                    T = Tc[1]
                elseif t <= tc[2]
                    T = minimum([Tc[1] + HR * (t - tc[1]), Tc[2]]);
                else
                    T = minimum([Tc[2] + HR * (t - tc[2]), Tc[3]]);
                end
            else
                @show "wrong i_exp $i_exp"
            end
        end
        return T
    end

    function richter2019!(du, u, p, t)

        R = 8.314E-3    # universal gas constant, kJ/mol*K

        # [Ea1-3, logA1-3, v3, T0, beta, i_exp]
        Ea = p[1:3]   # kJ/mol
        logA = p[4:6]
        v = p[7]
        
        T0 = p[8]
        beta = p[9]
        i_exp = p[10]

        u = clamp.(u, 1e-30, 1e2)
        cell = u[1]
        cella = u[2]
        char = u[3]

        T = getsampletemp(t, T0, beta, i_exp)

        k = (10 .^ logA) .* exp.(- Ea ./ R ./ T)

        w1 = k[1] * (cell)
        w2 = k[2] * (cella)
        w3 = k[3] * (cella)

        du[1] = - w1
        du[2] = w1 - (w2 + w3)
        du[3] = w3 * v
    end

    l_exp_data = [];
    l_exp_info = zeros(length(l_exp), 3);
    for (i_exp, value) in enumerate(l_exp)
        filename = string("exp_data/expdata_no" , string(value) , ".txt");
        exp_data = load_exp(filename);
        push!(l_exp_data, exp_data);
        T0 = exp_data[1, 2];  # initial temperature, K
        l_exp_info[i_exp, 1] = T0;
        l_exp_info[i_exp, 3] = value;
    end
    l_exp_info[:, 2] = readdlm("exp_data/beta.txt")[l_exp];

    function makeprob(i_exp)
        exp_data = l_exp_data[i_exp];
        tlist = exp_data[:, 1];
        tspan = (tlist[1], tlist[end]);
        p = vcat(Ea, logA, v, l_exp_info[i_exp, :]);
        prob = ODEProblem(richter2019!, u0, tspan, p);
        return prob, tlist
    end

    # Training
    function cost_singleexp(prob, exp_data)
        sol = solve(prob, alg=ode_solver,
                    reltol=1e-2, abstol=1e-4, saveat=exp_data[:, 1],
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
        return mean(l_loss)
    end

    # [Ea1-3, logA1-3, v3, T0, beta]
    # Ea = [243, 198, 153]  # kJ/mol   # 150
    # logA = [19.5, 14.1, 9.69]  # 15
    # v = 0.35  # R3

    Ea = ones(3) * 150;
    logA = ones(3) * 15;
    v = 0.5;
    u0 = [1.0, 0.0, 0.0];

end

# Main Process
function plot_sol(sol, exp_data, Tlist, cap, sol0=nothing)
    
    plt = plot(exp_data[:, 1]/60, exp_data[:, 3], seriestype = :scatter, label="exp");

    plot!(plt, sol.t/60, sum(sol, dims=1)', lw=2, label="model", legend=:best);
    
    if sol0 !== nothing
        plot!(plt, sol0.t/60, sum(sol0, dims=1)', label="initial model");
    end
    
    xlabel!(plt, "time [min]");
    ylabel!(plt, "mass");
    title!(plt, cap);

    plt2 = twinx();
    plot!(plt2, exp_data[:, 1]/60, Tlist, lw=2, ls=:dash, label="T", legend=:left);
    ylabel!(plt2, "T");
    
    return plt
end

for (i_exp, value) in enumerate(l_exp)

    @show i_exp, value

    prob, tlist = makeprob(i_exp);

    sol = solve(prob, alg=ode_solver,
                reltol=1e-3, abstol=1e-6,
                saveat=tlist);

    Tlist = copy(sol.t)
    for (i, t) in enumerate(sol.t)
        Tlist[i] = getsampletemp(t, prob.p[end-2], prob.p[end-1],prob.p[end])
    end

    plt = plot_sol(sol, l_exp_data[i_exp], Tlist, "initial_p_$value");

    savefig(plt, "fig/initial_p_$value");
end

# Multi-processing
function cost_function_parallel(p)
    function getcost(index)
        return cost_function(p[:, index])
    end
    return pmap(getcost, 1:size(p, 2))
end

lb_Ea = ones(3) * 0.;
ub_Ea = ones(3) * 300.0;
lb_logA = ones(3) * 0.;
ub_logA = ones(3) * 23.;
lb_v = ones(1) * 0;
ub_v = ones(1);
lb = vcat(lb_Ea, lb_logA, lb_v);
ub = vcat(ub_Ea, ub_logA, ub_v);
p0 = vcat(Ea, logA, v);

cost_function(p0)

using CMAEvolutionStrategy

# res = minimize(cost_function_parallel, p0, 0.2,
#                lower = lb, upper = ub,
#                verbosity = 1,
#                ftol = 1e-6,
#                popsize = 48,
#                parallel_evaluation = true,
#                multi_threading = true,
#                )

p = p0
f_min = 1000

for ii in 1:5
    println("optimization iter $ii")
    res = minimize(cost_function_parallel, p, 0.2,
                lower = lb, upper = ub,
                verbosity = 1,
                ftol = 1e-7,
                popsize = 48,
                noise_handling = CMAEvolutionStrategy.NoiseHandling(3),
                parallel_evaluation = true,
                multi_threading = true,
                )

    if fbest(res) < f_min
        f_min = fbest(res)
        p = xbest(res)
    end
end

# p = xbest(res)

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

    Tlist = copy(sol.t)
    for (i, t) in enumerate(sol.t)
        Tlist[i] = getsampletemp(t, prob.p[end-2],
                                prob.p[end-1],prob.p[end])
    end

    plt = plot_sol(sol, exp_data, Tlist, "optimized_p_$value", sol0)
    savefig(plt, "fig/optimized_p_$value")
end