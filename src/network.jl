np = nr * (ns + 4) + 1
p = randn(Float64, np) .* 1.e-2;
p[1:nr] .+= 0.8;  # w_b
p[nr*(ns+1)+1:nr*(ns+2)] .+= 0.8;  # w_out
p[nr*(ns+2)+1:nr*(ns+4)] .+= 0.1;  # w_b | w_Ea
p[end] = 0.1;  # slope

function p2vec(p)
    slope = p[end] .* 1.e1
    w_b = p[1:nr] .* (slope * 10.0)
    w_b = clamp.(w_b, 0, 50)

    w_out = reshape(p[nr+1:nr*(ns+1)], ns, nr)
    # i = 1
    # for j = 1:ns-2
    #     w_out[j, i] = -1.0
    #     i = i + 1
    # end
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
    w_in, w_b, w_out = p2vec(p)
    println("\n species (column) reaction (row)")
    println("w_in | Ea | b | n_ocen | lnA | w_out")
    show(stdout, "text/plain", round.(hcat(w_in', w_b, w_out'), digits = 2))
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
    logX = @. log(clamp(u, lb, 10.0))
    T = getsampletemp(t, T0, beta)
    w_in_x = w_in' * vcat(logX, R / T, log(T), ocen)
    du .= w_out * (@. exp(w_in_x + w_b))
end

tspan = [0.0, 1.0];
u0 = zeros(ns);
u0[1] = 1.0;
prob = ODEProblem(crnn!, u0, tspan, p, abstol = lb)

alg = AutoTsit5(TRBDF2(autodiff = true));
# sense = BacksolveAdjoint(checkpointing=true; autojacvec=ZygoteVJP());
sense = ForwardSensitivity(autojacvec = true)
function pred_n_ode(p, i_exp, exp_data)
    global T0, beta, ocen = l_exp_info[i_exp, :]
    global w_in, w_b, w_out = p2vec(p)
    ts = @view(exp_data[:, 1])
    tspan = [ts[1], ts[end]]
    sol = solve(
        prob,
        alg,
        tspan = tspan,
        p = p,
        saveat = @view(exp_data[:, 1]),
        sensalg = sense,
        maxiters = maxiters,
    )

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
@time loss = loss_neuralode(p, 1)
# using BenchmarkTools
# @benchmark loss = loss_neuralode(p, 1)
# @benchmark grad = ForwardDiff.gradient(x -> loss_neuralode(x, 1), p)
