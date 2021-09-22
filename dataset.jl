
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
    filename = string("exp_data/expdata_no", string(value), ".txt")
    exp_data = Float64.(load_exp(filename))

    if value == 4
        exp_data = exp_data[1:60, :]
    elseif value == 5
        exp_data = exp_data[1:58, :]
    elseif value == 6
        exp_data = exp_data[1:60, :]
    elseif value == 7
        exp_data = exp_data[1:71, :]
    end

    push!(l_exp_data, exp_data)
    l_exp_info[i_exp, 1] = exp_data[1, 2] # initial temperature, K
end
l_exp_info[:, 2] = readdlm("exp_data/beta.txt")[l_exp];
l_exp_info[:, 3] = log.(readdlm("exp_data/ocen.txt")[l_exp] .+ llb);
