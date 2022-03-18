module TCNN

using PyCall
using CUDA
using PyCallChainRules.Torch: TorchModuleWrapper

const torch = PyNULL()
const tcnn = PyNULL()

function network_with_input_encoding(; n_input_dims, n_output_dims, encoding_config, network_config)
    net = tcnn.NetworkWithInputEncoding(n_input_dims=n_input_dims, n_output_dims=n_output_dims, encoding_config=encoding_config, network_config=network_config)
    net = net.to(device=torch.device("cuda:0"))
    return TorchModuleWrapper(net)
end

function __init__()
    try
        copy!(torch, pyimport("torch"))
        copy!(tcnn, pyimport("tinycudann"))
    catch e
        @warn "Not installed correctly" e
    end
end

end

using .TCNN
using JSON
using CUDA
using Flux
using Optimisers
using Statistics

function loss(model, x, y)
    mean(abs2, (model(x) .- y))
end

function train()
    cfg = JSON.parsefile("/home/jagupt/src/tiny-cuda-nn/data/config_hash.json")
    n_channels = 3
    batchsize = 128
    nsteps = 100
    model = TCNN.network_with_input_encoding(n_input_dims=2, n_output_dims=n_channels, encoding_config=cfg["encoding"], network_config=cfg["network"])
    opt = Optimisers.ADAM(0.01)
    state = Optimisers.setup(opt, model)
    input = CUDA.randn(2, batchsize)
    target = CUDA.randn(3, batchsize)

    @info "before" loss(model, input, target)
    train_time = @elapsed begin
        for i in 1:nsteps
            gs, _ = Flux.gradient(model, input, target) do m, x, y
                loss(m, x, y)
            end
            state, model = Optimisers.update(state, model, gs)
        end
    end
    @info "after" loss(model, input, target)   
    @info "Took $(train_time)s" 
end