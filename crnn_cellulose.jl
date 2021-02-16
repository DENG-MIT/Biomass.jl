using Random, Plots
using Zygote, ForwardDiff
using OrdinaryDiffEq, DiffEqSensitivity
using LinearAlgebra
using Statistics
using ProgressBars, Printf
using Flux
using Flux.Optimise:update!
using Flux.Losses:mae
using BSON: @save, @load
using DelimitedFiles

is_restart = true
n_epoch = 5000;
n_plot = 10;
grad_max = 1.e1;
maxiters = 5000;

ns = 5;
nr = 5;

lb = 1.e-6;
llb = 1.e-6;
ub = 1.e1;

u0 = zeros(ns);
u0[1] = 1.0

const l_exp = 1:14
n_exp = length(l_exp)

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

opt = Flux.Optimiser(ExpDecay(5e-3, 0.2, length(l_train) * 500, 1e-4),
                     ADAMW(0.005, (0.9, 0.999), 1.e-8));

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
    exp_data = Float64.(load_exp(filename));

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

np = nr * (ns + 4) + 1
p = randn(Float64, np) .* 1.e-2;
p[1:nr] .+= 0.8;  # w_b
p[nr * (ns + 1) + 1:nr * (ns + 2)] .+= 0.8;  # w_out
p[nr * (ns + 2) + 1:nr * (ns + 4)] .+= 0.1;  # w_b | w_Ea
p[end] = 0.1;  # slope

function p2vec(p)
    slope = p[end] .* 1.e1
    w_b = p[1:nr] .* (slope * 10.0)
    w_b = clamp.(w_b, 0, 50)

    w_out = reshape(p[nr+1:nr*(ns+1)], ns, nr)
    i = 1
    for j = 1:ns-2
        w_out[j, i] = -1.0
        i = i + 1
    end
    @. w_out[1, :] = clamp(w_out[1, :], -3, 0)
    @. w_out[end, :] = clamp(abs(w_out[end, :]), 0, 3)
    w_out[ns-1:ns-1, :] .=
        -sum(w_out[1:ns-2, :], dims = 1) .- sum(w_out[ns:ns, :], dims = 1)

    w_in_Ea = abs.(p[nr*(ns+1)+1:nr*(ns+2)] .* (slope * 100.0))
    w_in_Ea = clamp.(w_in_Ea, 0.e0, 300.0)

    w_in_b = abs.(p[nr*(ns+2)+1:nr*(ns+3)])

    w_in_ocen = abs.(p[nr*(ns+3)+1:nr*(ns+4)])
    w_in_ocen = clamp.(w_in_ocen, 0.0, 1.5)
    # w_in_ocen[1:ns-1] .= 0

    w_in = vcat(clamp.(-w_out, 0.0, 4.0), w_in_Ea', w_in_b', w_in_ocen')
    return w_in, w_b, w_out
end

function display_p(p)
    w_in, w_b, w_out = p2vec(p);
    println("\n species (column) reaction (row)")
    println("w_in | Ea | b | n_ocen | lnA | w_out")
    show(stdout, "text/plain", round.(hcat(w_in', w_b, w_out'), digits=2))
    # println("\n w_out")
    # show(stdout, "text/plain", round.(w_out', digits=3))
    println("\n")
end
display_p(p);

function getsampletemp(t, T0, beta)
    if beta < 100
        T = T0 + beta / 60 * t  # K/min to K/s
    else
        tc = [999.0, 1059.0] .* 60.0
        Tc = [beta, 370.0, 500.0] .+ 273.0
        HR = 40.0 / 60.0
        if t <= tc[1]
            T = Tc[1]
        elseif t <= tc[2]
            T = minimum([Tc[1] + HR * (t - tc[1]), Tc[2]])
        else
            T = minimum([Tc[2] + HR * (t - tc[2]), Tc[3]])
        end
    end
    return T
end

const R = -1.0 / 8.314e-3  # universal gas constant, kJ/mol*K
@inbounds function crnn!(du, u, p, t)
    logX = @. log(clamp(u, llb, ub))
    T = getsampletemp(t, T0, beta)
    w_in_x = w_in' * vcat(logX, R / T, log(T), ocen)
    du .= w_out * (@. exp(w_in_x + w_b))
end

tspan = [0.0, 1.0];
prob = ODEProblem(crnn!, u0, tspan, p, abstol=lb)

alg = AutoTsit5(TRBDF2(autodiff=true));
# sense = BacksolveAdjoint(checkpointing=true; autojacvec=ZygoteVJP());
sense = ForwardSensitivity(autojacvec = true)
function pred_n_ode(p, i_exp, exp_data)
    global T0, beta, ocen = l_exp_info[i_exp, :]
    global w_in, w_b, w_out = p2vec(p)
    ts = @view(exp_data[:, 1])
    tspan = [ts[1], ts[end]]
    sol = solve(prob, alg, tspan=tspan, p = p,
                saveat = @view(exp_data[:, 1]),
                sensalg = sense, maxiters = maxiters,)

    if sol.retcode == :Success
        nothing
    else
        @sprintf("solver failed beta: %.0f ocen: %.2f", beta, exp(ocen))
    end
    return sol
end

function loss_neuralode(p, i_exp)
    exp_data = l_exp_data[i_exp]
    pred = Array(pred_n_ode(p, i_exp, exp_data))
    masslist = sum(clamp.(@view(pred[1:end-1, :]), 0, Inf), dims = 1)'
    gaslist = clamp.(@views(pred[end, :]), 0, Inf)
    loss = mae(masslist, @view(exp_data[1:length(masslist), 3]))
    if ocen < 1000.0
        loss += mae(gaslist, 1 .- @view(exp_data[1:length(masslist), 3]))
    end
    return loss
end
loss = loss_neuralode(p, 1)
# using BenchmarkTools
# @benchmark loss = loss_neuralode(p, 1)
# @benchmark grad = ForwardDiff.gradient(x -> loss_neuralode(x, 1), p)

include("callback.jl")

epochs = ProgressBar(iter:n_epoch);
loss_epoch = zeros(Float64, n_exp);
grad_norm = zeros(Float64, n_exp);
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
                    string(@sprintf("Loss train: %.2e val: %.2e grad: %.2e lr: %.1e",
                            loss_train, loss_val, grad_mean, opt[1].eta)))
    cb(p, loss_train, loss_val, grad_mean)
end

@sprintf("Min Loss train: %.2e val: %.2e", minimum(l_loss_train), minimum(l_loss_val))

for i_exp in randperm(n_exp)
    cbi(p, i_exp)
end
