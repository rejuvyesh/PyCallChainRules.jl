module TorchMLP
using PyCallChainRules.Torch: TorchModuleWrapper, torch, functorch, ispysetup

using Statistics
using BenchmarkTools

using Zygote

if !ispysetup[]
    return
end

include("pytorch_utils.jl")

batchsizes = [1, 8, 16, 32]
indim = 8
outdim = 4
hiddendim = 64

suite = BenchmarkGroup()

for batchsize in batchsizes
    mlp = torch.nn.Sequential(torch.nn.Linear(indim, hiddendim), torch.nn.ReLU(), torch.nn.Linear(hiddendim, outdim))

    s = suite["bs=$batchsize"] = BenchmarkGroup()
    # Forward pass
    s["forward"] = BenchmarkGroup()
    forward_pass!(s["forward"], mlp, randn(Float32, indim), batchsize)

    # Gradient pass
    s["backward"] = BenchmarkGroup()
    backward_pass!(s["backward"], mlp, randn(Float32, indim), batchsize)
end
end

TorchMLP.suite

