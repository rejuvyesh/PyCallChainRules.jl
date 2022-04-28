using PyCallChainRules.Torch: TorchModuleWrapper, torch, functorch, ispysetup, pyfrom_dlpack

using Test
using Zygote
using Flux
using ChainRulesCore: NoTangent, AbstractZero
import Random
using PyCall
using Functors: fmap
using DLPack
using CUDA

if !ispysetup[]
    return
end

fexp = pyimport("functorch.experimental")

if CUDA.functional()
    device = torch.device("cuda:0")
else
    device = torch.device("cpu")
end


model = torch.hub.load("pytorch/vision", "resnet18", pretrained=true)
model_gn = fexp.replace_all_batch_norm_modules_(model).to(device=device)
modelwrap = TorchModuleWrapper(model_gn)
if CUDA.functional()
    modelwrap = fmap(CUDA.cu, modelwrap)
end
x = randn(Float32, reverse((1, 3, 224, 224)))
if CUDA.functional()
    x = CUDA.cu(x)
end

grad,  = Zygote.gradient(m->sum(m(x)), modelwrap)
@test length(grad.params) == length(modelwrap.params)
params = map(x ->  DLPack.share(x, PyObject, pyfrom_dlpack).to(device = device, dtype = modelwrap.dtype).requires_grad_(true), modelwrap.params)
torch_out = modelwrap.torch_stateless_module(params, modelwrap.buffers, map(z-> DLPack.share(z, PyObject, pyfrom_dlpack).to(dtype=modelwrap.dtype, device=device), [x])...).sum()
torchgrad = map(x-> x.cpu().numpy(), torch.autograd.grad(torch_out, params))
@test length(torchgrad) == length(grad.params)
for i in 1:length(grad.params)
    @test isapprox(sum(torchgrad[i]), sum(grad.params[i]), atol=1e-3, rtol=1e-3)
end
