include("header.jl")
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

conf["loss_train"] = minimum(l_loss_train)
conf["loss_val"] = minimum(l_loss_val)
YAML.write_file(config_path, conf)

for i_exp in randperm(n_exp)
    cbi(p, i_exp)
end
