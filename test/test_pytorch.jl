using PyCallChainRules.Torch: TorchModuleWrapper, torch, functorch, dlpack, pyto_dlpack, pyfrom_dlpack, ispysetup

using Test
using Zygote
using Flux
using ChainRulesCore: NoTangent, AbstractZero
import Random
using PyCall
using CUDA
using DLPack

if !ispysetup[]
    return
end

if CUDA.functional()
    device = torch.device("cuda:0")
else
    device = torch.device("cpu")
end

function compare_grad_wrt_params(modelwrap, inputs...)
    params = map(x -> DLPack.share(x, PyObject, pyfrom_dlpack).to(device = device, dtype = modelwrap.dtype).requires_grad_(true), (modelwrap.params))
    torch_out = modelwrap.torch_stateless_module(params, modelwrap.buffers, map(z->DLPack.share(z, PyObject, pyfrom_dlpack).to(dtype=modelwrap.dtype, device=device), inputs)...).sum()
    torchgrad = map(x-> (x.cpu().numpy()), torch.autograd.grad(torch_out, params))
    grad,  = Zygote.gradient(m->sum(m(inputs...)), modelwrap)
    @test length(torchgrad) == length(grad.params)
    for i in 1:length(grad.params)
        @test isapprox(sum(torchgrad[i]), sum(grad.params[i]))
    end
    @test length(grad.params) == length(modelwrap.params)
    @test grad.params[1] !== nothing
    @test grad.params[2] !== nothing
    @test size(grad.params[1]) == size(modelwrap.params[1])
    @test size(grad.params[2]) == size(modelwrap.params[2])
end

function compare_grad_wrt_inputs(modelwrap, x)
    params = map(z -> DLPack.share(z, PyObject, pyfrom_dlpack).to(device = device, dtype = modelwrap.dtype).requires_grad_(true), (modelwrap.params))
    xtorch = DLPack.share(copy(x), PyObject, pyfrom_dlpack).to(dtype=modelwrap.dtype, device=device).requires_grad_(true)
    torch_out = modelwrap.torch_stateless_module(params, modelwrap.buffers, xtorch).sum()
    torchgrad = map(z-> (copy(z.cpu().numpy())), torch.autograd.grad(torch_out, xtorch))[1]
    grad, = Zygote.gradient(z->sum(modelwrap(z)), x)
    @test size(grad) == size(x)
    @test length(torchgrad) == length(grad)
    @test isapprox(sum(torchgrad), sum(grad))
end


@testset "dlpack" begin
    for dims in ((10,), (1, 10), (2, 3, 5), (2, 3, 4, 5))
        xto = torch.randn(dims..., device=device)
        xjl = DLPack.wrap(xto, pyto_dlpack)
        @test Tuple(xto.size()) == reverse(size(xjl))
        @test isapprox(sum(xto.cpu().numpy()), sum(xjl))
    end
end

batchsize = 1
indim = 3
outdim = 2
hiddendim = 4

@testset "linear" begin
    lin = torch.nn.Linear(indim, outdim).to(device=device)
    linwrap = TorchModuleWrapper(lin)
    if CUDA.functional()
        linwrap = fmap(CUDA.cu, linwrap)
    end    
    x = randn(Float32, indim, batchsize)
    if CUDA.functional()
        x = cu(x)
    end
    y = linwrap(x)
    @test size(y) == (outdim, batchsize)
    compare_grad_wrt_params(linwrap, x)
    compare_grad_wrt_inputs(linwrap, x)

end

@testset "mlp" begin
    mlp = torch.nn.Sequential(torch.nn.Linear(indim, hiddendim), torch.nn.ReLU(), torch.nn.Linear(hiddendim, outdim)).to(device=device)
    mlpwrap = TorchModuleWrapper(mlp)
    if CUDA.functional()
        mlpwrap = fmap(CUDA.cu, mlpwrap)
    end    

    x = randn(Float32, indim, batchsize)
    if CUDA.functional()
        x = cu(x)
    end
    y = mlpwrap(x)
    @test size(y) == (outdim, batchsize)
    compare_grad_wrt_params(mlpwrap, x)
    compare_grad_wrt_inputs(mlpwrap, x)
end

# Flux check
@testset "flux" begin
    lin = torch.nn.Linear(indim, outdim).to(device=device)
    linwrap = TorchModuleWrapper(lin)
    nn = Chain(Dense(4, 3), linwrap)
    if CUDA.functional()
        nn = Flux.gpu(nn)
    end
    x2 = randn(Float32, 4, batchsize)
    if CUDA.functional()
        x2 = cu(x2)
    end
    grad,  = Zygote.gradient(m->sum(m(x2)), nn)
    @test grad !== nothing    
end


@testset "conv" begin
    model = torch.nn.Sequential(
            torch.nn.Conv2d(1,2,5),
            torch.nn.ReLU(),
            torch.nn.Conv2d(2,6,5),
            torch.nn.ReLU()
            ).to(device=device)
    modelwrap = TorchModuleWrapper(model)
    if CUDA.functional()
        modelwrap = fmap(CUDA.cu, modelwrap)
    end
    input = randn(Float32, 12, 12, 1, batchsize)
    if CUDA.functional()
        input = cu(input)
    end
    output = modelwrap(input)

    compare_grad_wrt_params(modelwrap, input)
    compare_grad_wrt_inputs(modelwrap, input)
end