
function plot_sol(i_exp, sol, exp_data, Tlist, cap, sol0 = nothing)
    T0, beta, ocen = l_exp_info[i_exp, :]
    ts = sol.t / 60.0
    ind = length(ts)
    plt = plot(
        exp_data[:, 1] / 60.0,
        exp_data[:, 3],
        seriestype = :scatter,
        label = "Exp",
    )

    plot!(
        plt,
        ts,
        sum(clamp.(sol[1:end-1, :], 0, Inf), dims = 1)',
        lw = 3,
        legend = :left,
        label = "CRNN",
    )

    if sol0 !== nothing
        plot!(
            plt,
            sol0.t / 60,
            sum(sol0[1:end-1, :], dims = 1)',
            label = "initial model",
        )
    end
    xlabel!(plt, "Time [min]")
    ylabel!(plt, "Mass")
    title!(plt, cap)
    exp_cond = string(
        @sprintf(
            "T0 = %.1f K \n beta = %.2f K/min \n [O2] = %.2f",
            T0,
            beta,
            exp(ocen) * 100.0
        )
    )
    annotate!(plt, exp_data[end, 1] / 60.0 * 0.85, 0.4, exp_cond)

    plt2 = twinx()
    plot!(
        plt2,
        exp_data[1:ind, 1] / 60,
        Tlist,
        lw = 2,
        ls = :dash,
        legend = :topright,
        label = "T",
    )
    ylabel!(plt2, "Temp")

    p2 = plot(ts, sol[1, :], lw = 2, legend = :right, label = "Cellulose")
    for i = 2:ns
        plot!(p2, ts, sol[i, :], lw = 2, label = "S$i")
    end
    xlabel!(p2, "Time [min]")
    ylabel!(p2, "Mass")

    plt = plot(plt, p2, framestyle = :box, layout = @layout [a; b])
    plot!(plt, size = (800, 800))
    return plt
end

cbi = function (p, i_exp)
    exp_data = l_exp_data[i_exp]
    sol = pred_n_ode(p, i_exp, exp_data)
    Tlist = similar(sol.t)
    T0, beta, ocen = l_exp_info[i_exp, :]
    for (i, t) in enumerate(sol.t)
        Tlist[i] = getsampletemp(t, T0, beta)
    end
    value = l_exp[i_exp]
    plt = plot_sol(i_exp, sol, exp_data, Tlist, "exp_$value")
    png(plt, string(fig_path, "/conditions/pred_exp_$value"))
    return false
end

function plot_loss(l_loss_train, l_loss_val; yscale = :log10)
    plt_loss = plot(l_loss_train, yscale = yscale, label = "train")
    plot!(plt_loss, l_loss_val, yscale = yscale, label = "val")
    plt_grad = plot(list_grad, yscale = yscale, label = "grad_norm")
    xlabel!(plt_loss, "Epoch")
    ylabel!(plt_loss, "Loss")
    xlabel!(plt_grad, "Epoch")
    ylabel!(plt_grad, "Gradient Norm")
    ylims!(plt_loss, (-Inf, 1e0))
    ylims!(plt_grad, (-Inf, 1e3))
    plt_all = plot([plt_loss, plt_grad]..., legend = :top, framestyle=:box)
    plot!(
        plt_all,
        size=(1000, 450),
        xtickfontsize = 11,
        ytickfontsize = 11,
        xguidefontsize = 12,
        yguidefontsize = 12,
    )
    png(plt_all, string(fig_path, "/loss_grad"))
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
        @printf(
            "Min Loss train: %.2e val: %.2e",
            minimum(l_loss_train),
            minimum(l_loss_val)
        )
        println("update plot ", l_exp[list_exp])
        for i_exp in list_exp
            cbi(p, i_exp)
        end

        plot_loss(l_loss_train, l_loss_val; yscale = :log10)

        @save string(ckpt_path, "/mymodel.bson") p opt l_loss_train l_loss_val list_grad iter
    end
    iter += 1
end

if is_restart
    @load string(ckpt_path, "/mymodel.bson") p opt l_loss_train l_loss_val list_grad iter
    iter += 1
end
