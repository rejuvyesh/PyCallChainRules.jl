module TorchHub
using PyCallChainRules.Torch: TorchModuleWrapper, torch, functorch, ispysetup

using Statistics
using BenchmarkTools

using Zygote
using PyCallChainRules.Torch.PyCall

using CUDA

if !ispysetup[]
    return
end

py"""
import torch
def bn2group(module):
    num_groups = 16 # hyper_parameter of GroupNorm
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.GroupNorm(num_groups,
                                           module.num_features,
                                           module.eps, 
                                           module.affine,
                                          )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(name, bn2group(child))

    del module
    return module_output
"""

include("pytorch_utils.jl")

batchsizes = [1, ]

suite = BenchmarkGroup()

for device in ["cpu", "cuda"]
    ss = suite["$device"] = BenchmarkGroup()
    for batchsize in batchsizes
        model = torch.hub.load("pytorch/vision", "resnet18", pretrained=true)
        model_gn = py"bn2group"(model)

        s = ss["bs=$batchsize"] = BenchmarkGroup()
        # Forward pass
        s["forward"] = BenchmarkGroup()
        forward_pass!(s["forward"], model_gn, randn(Float32, 3, 224, 224), batchsize, device; samples=20)

        # Gradient pass
        s["backward"] = BenchmarkGroup()    
        b2 = backward_pass!(s["backward"], model_gn, randn(Float32, 3, 224, 224), batchsize, device; samples=20)
    end
end
end
TorchHub.suite