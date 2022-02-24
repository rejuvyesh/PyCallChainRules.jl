module TorchMLP
using PyCallChainRules.Torch: TorchModuleWrapper, torch, functorch, ispysetup

using Statistics
using BenchmarkTools

using Zygote

using CUDA

if !ispysetup[]
    return
end

include("pytorch_utils.jl")

batchsizes = [1, 8, 16, 32]
indim = 8
outdim = 4
hiddendim = 64

suite = BenchmarkGroup()

for device in ["cpu"]
    ss = suite["$device"] = BenchmarkGroup()
    for batchsize in batchsizes
        mlp = torch.nn.Sequential(torch.nn.Linear(indim, hiddendim), torch.nn.ReLU(), torch.nn.Linear(hiddendim, outdim))

        s = ss["bs=$batchsize"] = BenchmarkGroup()
        # Forward pass
        s["forward"] = BenchmarkGroup()
        forward_pass!(s["forward"], mlp, randn(Float32, indim), batchsize, device)

        # Gradient pass
        s["backward"] = BenchmarkGroup()
        backward_pass!(s["backward"], mlp, randn(Float32, indim), batchsize, device)
    end
end

for device in ["cuda"]
    ss = suite["$device"] = BenchmarkGroup()
    for batchsize in batchsizes
        mlp = torch.nn.Sequential(torch.nn.Linear(indim, hiddendim), torch.nn.ReLU(), torch.nn.Linear(hiddendim, outdim))

        s = ss["bs=$batchsize"] = BenchmarkGroup()
        # Forward pass
        s["forward"] = BenchmarkGroup()
        forward_pass!(s["forward"], mlp, randn(Float32, indim), batchsize, device)

        # Gradient pass
        s["backward"] = BenchmarkGroup()
        backward_pass!(s["backward"], mlp, randn(Float32, indim), batchsize, device)
    end
end


end
TorchMLP.suite

