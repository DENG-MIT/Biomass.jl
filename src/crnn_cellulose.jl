using Random, Plots
using Zygote, ForwardDiff
using OrdinaryDiffEq, DiffEqSensitivity
using LinearAlgebra
using Statistics
using ProgressBars, Printf
using Flux
using Flux.Optimise: update!
using Flux.Losses: mae
using BSON: @save, @load
using DelimitedFiles

is_restart = false
n_epoch = 5000;
n_plot = 10;
grad_max = 1.e1;
maxiters = 5000;

ns = 5;
nr = 5;

lb = 1.e-8;
llb = lb;

u0 = zeros(ns);
u0[1] = 1.0

const l_exp = 1:14
n_exp = length(l_exp)

l_train = []
l_val = []
for i = 1:n_exp
    j = l_exp[i]
    if !(j in [2, 6, 9, 12])
        push!(l_train, i)
    else
        push!(l_val, i)
    end
end

opt = Flux.Optimiser(
    ExpDecay(5e-3, 0.2, length(l_train) * 500, 1e-4),
    ADAMW(0.005, (0.9, 0.999), 1.e-8),
);

include("dataset.jl")
include("network.jl")
include("callback.jl")

# opt = ADAMW(1.e-6, (0.9, 0.999), 1.e-8);

epochs = ProgressBar(iter:n_epoch);
loss_epoch = zeros(Float64, n_exp);
grad_norm = zeros(Float64, n_exp);
for epoch in epochs
    global p
    for i_exp in randperm(n_exp)
        if i_exp in l_val
            continue
        end
        grad = ForwardDiff.gradient(x -> loss_neuralode(x, i_exp), p)
        grad_norm[i_exp] = norm(grad, 2)
        if grad_norm[i_exp] > grad_max
            grad = grad ./ grad_norm[i_exp] .* grad_max
        end
        update!(opt, p, grad)
    end
    for i_exp = 1:n_exp
        loss_epoch[i_exp] = loss_neuralode(p, i_exp)
    end
    loss_train = mean(loss_epoch[l_train])
    loss_val = mean(loss_epoch[l_val])
    grad_mean = mean(grad_norm[l_train])
    set_description(
        epochs,
        string(
            @sprintf(
                "Loss train: %.2e val: %.2e grad: %.2e lr: %.1e",
                loss_train,
                loss_val,
                grad_mean,
                opt[1].eta
            )
        ),
    )
    cb(p, loss_train, loss_val, grad_mean)
end

@sprintf(
    "Min Loss train: %.2e val: %.2e",
    minimum(l_loss_train),
    minimum(l_loss_val)
)

for i_exp in randperm(n_exp)
    cbi(p, i_exp)
end
