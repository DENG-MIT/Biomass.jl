function getMLR(t, m)
    mlr = copy(m)
    mlr[2:end] = (m[2:end] .- m[1:end-1]) ./ (t[2:end] .- t[1:end-1])
    mlr[1] = mlr[2]
    return - mlr .* 60.0
end

# Plot TGA MLR data
list_plt = []
for i_exp = 1:14
    T0, beta, ocen = l_exp_info[i_exp, :]
    exp_data = l_exp_data[i_exp]
    sol = pred_n_ode(p, i_exp, exp_data)
    Tlist = similar(sol.t)
    T0, beta, ocen = l_exp_info[i_exp, :]
    for (i, t) in enumerate(sol.t)
        Tlist[i] = getsampletemp(t, T0, beta)
    end

    conc_end[i_exp, :] .= sol[:, end]

    if beta < 100
        plt = plot(
            Tlist,
            getMLR(exp_data[:, 1], exp_data[:, 3]),
            seriestype = :scatter,
            label = "Exp-$i_exp",
            framestyle = :box,
        )
        plot!(
            plt,
            Tlist,
            getMLR(sol.t, sum(sol[1:end-1, :], dims = 1)'),
            lw = 2,
            legend = false,
        )
        xlabel!(plt, "Temperature [K]")

    else
        plt = plot(
            exp_data[:, 1] / 60.0,
            getMLR(exp_data[:, 1], exp_data[:, 3]),
            seriestype = :scatter,
            label = "Exp-$i_exp",
            framestyle = :box,
        )
        plot!(
            plt,
            sol.t / 60,
            getMLR(sol.t, sum(sol[1:end-1, :], dims = 1)'),
            lw = 2,
            legend = false,
        )
        xlabel!(plt, "Time [min]")
    end

    if i_exp in [4, 5, 6, 7]
        xlims!(plt, (0.0, 900))
        ylims!(plt, (0.0, 0.008))
    end

    exp_cond = @sprintf("T0:%.0f K", T0)
    if beta < 100.0
        exp_cond *= @sprintf("\nbeta:%.0f K/min", beta)
    end
    if exp(ocen) * 100.0 > 0.001
        exp_cond *= @sprintf("\n[O2]:%.2f", exp(ocen))
    else
        exp_cond *= @sprintf("\ninert")
    end

    ann_loc = [0.2, 0.5]
    if i_exp in [6, 7, 12, 13, 14]
        ann_loc = [0.7, 0.6]
    end
    annotate!(plt, xlims(plt)[1] + (xlims(plt)[end] - xlims(plt)[1]) * ann_loc[1],
                   ylims(plt)[1] + (ylims(plt)[end] - ylims(plt)[1]) * ann_loc[2],
                   text(exp_cond, 11))

    ylabel!(plt, "MLR [-] (No. $i_exp)")
    # ylims!(plt, (0.0, 1.0))
    plot!(
        plt,
        xtickfontsize = 11,
        ytickfontsize = 11,
        xguidefontsize = 12,
        yguidefontsize = 12,
    );

    push!(list_plt, plt);
end
plt_all = plot(list_plt..., layout = (7, 2));
plot!(plt_all, size = (800, 1200));
png(plt_all, string(fig_path, "/TGA_MLR_summary"))