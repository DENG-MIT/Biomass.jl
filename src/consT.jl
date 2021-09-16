@inbounds function crnn!(du, u, p, t)
    logX = @. log(clamp(u, lb, 10.0))
    T = T0 + 273.15
    w_in_x = w_in' * vcat(logX, R / T, log(T), ocen)
    du .= w_out * (@. exp(w_in_x + w_b))
end

# function crnn!(du, u, p, t)
#     # T = 500.0 .+ 10.0 * t/60.0
#     T = T0 + 273.15
#     u_ = clamp.(u, lb, 10.0)
#     r1 = exp(22.69 + 0.05 * log(T) + 88.7 * R/T) * u_[3]^1.32 * ocen^0.45
#     r2 = exp(17.3 + 222.4 * R/T) * u_[1]^0.2
#     r3 = exp(33.64 + 0.04 * log(T) + 187.4 * R/T) * u_[1]^1.52 * ocen^0.19
#     r4 = exp(14.92 + 0.15 * log(T) + 117.2 * R/T) * u_[1]^0.4 * u_[3]^0.61
#     r5 = exp(36.75 + 0.34 * log(T) + 218.0 * R/T) * u_[1]^1.15 * u_[2]^0.38
#     r6 = exp(14.08 + 0.03 * log(T) + 110.5 * R/T) * u_[2]^1.91 * ocen^0.33
#     du[1] = -0.2 * r2 -1.52 * r3 - 0.4 * r4 - 1.15 * r5
#     du[2] = 0.56 * r1 + 0.41 * r3 + 0.41 * r4 - 0.38 * r5 - 1.91 * r6
#     du[3] = -1.32 * r1 + 0.2 * r2 + 0.46 * r3 - 0.61 * r4 + 0.64 * r5 + 1.27 * r6
#     du[4] = 0.76 * r1 + 0.68* r3 + 0.61 * r4 + 0.89 * r5 + 0.63 * r6
# end

tspan = [0.0, 100.0 * 60.0];
u0 = zeros(ns);
u0[1] = 1.0;
prob = ODEProblem(crnn!, u0, tspan, p, abstol = lb/100.0)
ocen = llb
w_in, w_b, w_out = p2vec(p)

T0 = 500.0

using Sundials

condition(u, t, integrator) = u[1] < lb * 10.0
affect!(integrator) = terminate!(integrator)
_cb = DiscreteCallback(condition, affect!)

sol = solve(
    prob,
    CVODE_BDF(),
    tspan = tspan,
    p = p,
    saveat = [],
    # callback = _cb,
)

varnames = ["Cellu", "S2", "S3", "Vola"]
xL = 1.e-2
plt = plot(sol.t .+ xL, clamp.(sol[1, :], -1, Inf), label = varnames[1])
for i in 2:ns
    scale_factor = 1 ./ maximum(sol[:, :], dims=2)
    scale_factor .= 1.0
    if scale_factor[i] > 1.1
        _label =  @sprintf("%s x %.2e", varnames[i], scale_factor[i])
    else
        _label = varnames[i]
    end
    plot!(plt, sol.t .+ xL, clamp.(sol[i, :], 0, Inf) * scale_factor[i], label = _label)
end
xlabel!(plt, "Time [min]");
ylabel!(plt, "Mass (-)");
plot!(plt, size=(500, 500), legend=:outerright, framestyle=:box, lw=3)
plot!(
    plt,
    xscale=:log10,
    xtickfontsize = 11,
    ytickfontsize = 11,
    xguidefontsize = 12,
    yguidefontsize = 12,
)
title!(plt, @sprintf("Final @%.0f C [Cellu, S2, S3, Vola] = \n %.1e %.1e %.1e %.1e",
     T0, sol[1, end], sol[2, end], sol[3, end], sol[4, end]))

png(plt, string(fig_path, "/pred_S_exp_$T0 C"));
