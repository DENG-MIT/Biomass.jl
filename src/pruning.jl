include("header.jl")
include("dataset.jl")
include("network.jl")
include("callback.jl")

# pyplot()
plot_loss(l_loss_train, l_loss_val; yscale = :log10)

function loss_neuralode_res(p)
    l_loss_exp = zeros(n_exp)
    for i_exp in 1:n_exp
        exp_data = l_exp_data[i_exp]
        pred = Array(pred_n_ode(p, i_exp, exp_data))
        masslist = sum(clamp.(@view(pred[1:end-1, :]), 0, Inf), dims = 1)'
        l_loss_exp[i_exp] = mae(masslist, @view(exp_data[1:length(masslist), 3]))
    end
    return l_loss_exp
end

println("results after pruning")
maxiters = 1e5
p_cutoff = 0.185
loss_epoch = zeros(Float64, n_exp);
for i_exp = 1:n_exp
    loss_epoch[i_exp] = loss_neuralode(p, i_exp)
end
loss_train = mean(loss_epoch[l_train])
loss_val = mean(loss_epoch[l_val])
@printf(
    "Loss train: %.2f val: %.2f p_cutoff: %.2e",
    log10(loss_train),
    log10(loss_val),
    p_cutoff
)

display_p(p)
for i_exp in randperm(n_exp)
    cbi(p, i_exp)
end

l_loss_exp = loss_neuralode_res(p)

l_exp_data = []
l_exp_info = zeros(Float64, length(l_exp), 3)
for (i_exp, value) in enumerate(l_exp)
    filename = string("exp_data/expdata_no", string(value), ".txt")
    exp_data = Float64.(load_exp(filename))
    push!(l_exp_data, exp_data)
    l_exp_info[i_exp, 1] = exp_data[1, 2] # initial temperature, K
end
l_exp_info[:, 2] = readdlm("exp_data/beta.txt")[l_exp]
l_exp_info[:, 3] = log.(readdlm("exp_data/ocen.txt")[l_exp] .+ llb)

l_loss_exp = loss_neuralode_res(p)

conc_end = zeros(n_exp, ns)

# Plot TGA data
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
            exp_data[:, 3],
            seriestype = :scatter,
            label = "Exp-$i_exp",
            framestyle = :box,
        )
        plot!(
            plt,
            Tlist,
            sum(sol[1:end-1, :], dims = 1)',
            lw = 2,
            legend = false,
        )
        xlabel!(plt, "Temperature [K]")

    else
        plt = plot(
            exp_data[:, 1] / 60.0,
            exp_data[:, 3],
            seriestype = :scatter,
            label = "Exp-$i_exp",
            framestyle = :box,
        )
        plot!(
            plt,
            sol.t / 60,
            sum(sol[1:end-1, :], dims = 1)',
            lw = 2,
            legend = false,
        )
        xlabel!(plt, "Time [min]")
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

    ann_loc = [0.2, 0.3]
    if i_exp in [6, 7, 14]
        ann_loc = [0.7, 0.4]
    end
    annotate!(plt, xlims(plt)[1] + (xlims(plt)[end] - xlims(plt)[1]) * ann_loc[1],
                   ylims(plt)[1] + (ylims(plt)[end] - ylims(plt)[1]) * ann_loc[2],
                   text(exp_cond, 11))

    ylabel!(plt, "Mass [-] (No. $i_exp)")
    ylims!(plt, (0.0, 1.0))
    plot!(
        plt,
        xtickfontsize = 11,
        ytickfontsize = 11,
        xguidefontsize = 12,
        yguidefontsize = 12,
    )

    push!(list_plt, plt)
end
plt_all = plot(list_plt..., layout = (7, 2))
plot!(plt_all, size = (800, 1200))
png(plt_all, string(fig_path, "/TGA_mass_summary"))

# https://github.com/DENG-MIT/Biomass.jl/blob/main/backup/crnn_cellulose_ocen_test.jl
varnames = ["Cellu", "S2", "S3", "Vola"]
for i_exp in 1:n_exp
    T0, beta, ocen = l_exp_info[i_exp, :]
    exp_data = l_exp_data[i_exp]
    sol = pred_n_ode(p, i_exp, exp_data)
    Tlist = similar(sol.t)
    T0, beta, ocen = l_exp_info[i_exp, :]
    for (i, t) in enumerate(sol.t)
        Tlist[i] = getsampletemp(t, T0, beta)
    end
    value = l_exp[i_exp]
    list_plt = []
    scale_factor = 1 ./ maximum(sol[:, :], dims=2)
    scale_factor .= 1.0
    plt = plot(Tlist, clamp.(sol[1, :], 0, Inf), label = varnames[1])
    for i in 2:ns
        if scale_factor[i] > 1.1
            _label =  @sprintf("%s x %.2e", varnames[i], scale_factor[i])
        else
            _label = varnames[i]
        end
        plot!(plt, Tlist, clamp.(sol[i, :], 0, Inf) * scale_factor[i], label = _label)
    end
    xlabel!(plt, "Temperature [K]");
    ylabel!(plt, "Mass (-)");
    plot!(plt, size=(350, 350), legend=:topleft, framestyle=:box)
    plot!(
        plt,
        xtickfontsize = 11,
        ytickfontsize = 11,
        xguidefontsize = 12,
        yguidefontsize = 12,
    )
    png(plt, string(fig_path, "/pred_S_exp_$value"))
end

include("dataset.jl")
gr()
