using PyCallChainRules.Torch: TorchModuleWrapper, torch, functorch, ispysetup, pyfrom_dlpack

using Test
using ChainRulesTestUtils
using Zygote
using Flux
using ChainRulesCore: NoTangent, AbstractZero
import Random
using PyCall
using Functors: fmap
using CUDA

if !ispysetup[]
    return
end
if CUDA.functional()
    device = torch.device("cuda:0")
else
    device = torch.device("cpu")
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

model = torch.hub.load("pytorch/vision", "resnet18", pretrained=true)
model_gn = py"bn2group"(model).to(device=device)
modelwrap = TorchModuleWrapper(model_gn)
if CUDA.functional()
    modelwrap = fmap(CUDA.cu, modelwrap)
end
x = randn(Float32, reverse((1, 3, 224, 224)))
if CUDA.functional()
    x = CUDA.cu(x)
end
#y = modelwrap(x)

grad,  = Zygote.gradient(m->sum(m(x)), modelwrap)
@test length(grad.params) == length(modelwrap.params)
params = map(x ->  DLPack.share(x, pyfrom_dlpack).to(device = device, dtype = modelwrap.dtype).requires_grad_(true), modelwrap.params)
torch_out = modelwrap.torch_stateless_module(params, modelwrap.buffers, map(z-> DLPack.share(z, pyfrom_dlpack).to(dtype=modelwrap.dtype, device=device), [x])...).sum()
torchgrad = map(x-> x.cpu().numpy(), torch.autograd.grad(torch_out, params))
@test length(torchgrad) == length(grad.params)
for i in 1:length(grad.params)
    @test isapprox(sum(torchgrad[i]), sum(grad.params[i]))
end
