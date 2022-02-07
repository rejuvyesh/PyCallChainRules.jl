using PyCallChainRules: ReverseDimsArray

using PyCallChainRules.Torch: TorchModuleWrapper, torch, functorch, dlpack, pyto_dlpack, ispysetup

using Test
using Zygote
using Flux
using ChainRulesCore: NoTangent, AbstractZero
import Random
using PyCall
using DLPack

if !ispysetup[]
    return
end

function compare_grad_wrt_params(modelwrap, inputs...)
    params = map(x -> torch.as_tensor(copy(ReverseDimsArray(x))).to(device = modelwrap.device, dtype = modelwrap.dtype).requires_grad_(true), (modelwrap.params))
    torch_out = modelwrap.torch_stateless_module(params, modelwrap.buffers, map(z->torch.as_tensor(PyReverseDims(copy(z))).to(dtype=modelwrap.dtype), inputs)...).sum()
    torchgrad = map(x-> ReverseDimsArray(x.numpy()), torch.autograd.grad(torch_out, params))
    grad,  = Zygote.gradient(m->sum(m(inputs...)), modelwrap)
    @test length(torchgrad) == length(grad.params)
    for i in 1:length(grad.params)
        @test isapprox(torchgrad[i], grad.params[i])
    end
    @test length(grad.params) == length(modelwrap.params)
    @test grad.params[1] !== nothing
    @test grad.params[2] !== nothing
    @test size(grad.params[1]) == size(modelwrap.params[1])
    @test size(grad.params[2]) == size(modelwrap.params[2])
end

function compare_grad_wrt_inputs(modelwrap, x)
    params = map(z -> torch.as_tensor(copy(ReverseDimsArray(z))).to(device = modelwrap.device, dtype = modelwrap.dtype).requires_grad_(true), (modelwrap.params))
    xtorch = torch.as_tensor(PyReverseDims(copy(x))).to(dtype=modelwrap.dtype).requires_grad_(true)
    torch_out = modelwrap.torch_stateless_module(params, modelwrap.buffers, xtorch).sum()
    torchgrad = map(z-> ReverseDimsArray(copy(z.numpy())), torch.autograd.grad(torch_out, xtorch))[1]
    grad, = Zygote.gradient(z->sum(modelwrap(z)), copy(x))
    @test size(grad) == size(x)
    @test length(torchgrad) == length(grad)
    @test isapprox(torchgrad, grad)
end

# Random.seed!(42)
# torch.manual_seed(42)

@testset "dlpack" begin
    for dims in ((10,), (1, 10), (2, 3, 5), (2, 3, 4, 5))
        xto = torch.randn(dims...)
        xjl = DLArray(xto, pyto_dlpack)
        @test isapprox(xto.numpy(), xjl)
    end
end

batchsize = 1
indim = 3
outdim = 2
hiddendim = 4

@testset "linear" begin
    lin = torch.nn.Linear(indim, outdim)
    torchparams = Tuple([copy(DLArray(p, pyto_dlpack)) for p in lin.parameters()]) # (outdim, indim), (outdim,)),
    linwrap = TorchModuleWrapper(lin)
    # for i in 1:length(torchparams)
    #     @test isapprox(torchparams[i], linwrap.params[i])
    # end
    x = randn(Float32, indim, batchsize)
    y = linwrap(x)

    # for i in 1:length(torchparams)
    #     @test isapprox(torchparams[i], linwrap.params[i] )
    # end
    @test size(y) == (outdim, batchsize)
    compare_grad_wrt_params(linwrap, deepcopy(x))
    # for i in 1:length(torchparams)
    #     @test isapprox(torchparams[i], linwrap.params[i])
    # end
    compare_grad_wrt_inputs(linwrap, deepcopy(x))
    # for i in 1:length(torchparams)
    #     @test isapprox(torchparams[i], linwrap.params[i])
    # end
end

@testset "mlp" begin
    mlp = torch.nn.Sequential(torch.nn.Linear(indim, hiddendim), torch.nn.ReLU(), torch.nn.Linear(hiddendim, outdim))
    torchparams = Tuple([copy(DLArray(p, pyto_dlpack)) for p in mlp.parameters()])
    mlpwrap = TorchModuleWrapper(mlp)
    # for i in 1:length(torchparams)
    #     @test isapprox(torchparams[i], mlpwrap.params[i])
    # end

    x = randn(Float32, indim, batchsize)
    y = mlpwrap(x)
    # for i in 1:length(torchparams)
    #     @test isapprox(torchparams[i], mlpwrap.params[i])
    # end

    @test size(y) == (outdim, batchsize)
    compare_grad_wrt_params(mlpwrap, deepcopy(x))
    # for i in 1:length(torchparams)
    #     @test isapprox(torchparams[i], mlpwrap.params[i])
    # end

    compare_grad_wrt_inputs(mlpwrap, deepcopy(x))
    # for i in 1:length(torchparams)
    #     @test isapprox(torchparams[i], mlpwrap.params[i])
    # end

end
# CRTU check
# x = randn(Float32, indim, batchsize)
# test_rrule(linwrap, x; check_inferred=false, check_thunked_output_tangent=false, atol=1e-4, rtol=1e-4)
# const CRTU = ChainRulesTestUtils
# primals_and_tangents = CRTU.auto_primal_and_tangent((linwrap, x))
# CRTU.primal(primals_and_tangents)
# CRTU.tangent(primals_and_tangents)
# primals = CRTU.primal(primals_and_tangents)
# accum_cotangents = CRTU.tangent(primals_and_tangents)
# using ChainRulesCore: rrule
# config = CRTU.ADviaRuleConfig()
# res = rrule(config, primals...)
# y_ad, pullback = res
# call(f, xs...) = f(xs...;)
# call(primals...)
# y = call(primals...)
# ȳ = CRTU.rand_tangent(y)
# ad_cotangents = pullback(ȳ)
# length(primals)
# length(ad_cotangents)
# ad_cotangents[1]
# ad_cotangents[2]
# is_ignored = isa.(accum_cotangents, NoTangent)
# fd_cotangents =  CRTU._make_j′vp_call(CRTU._fdm, call, ȳ, primals, is_ignored)
#call2 = CRTU._wrap_function(call, primals, is_ignored)
#CRTU.test_approx(ad_cotangents[1], fd_cotangents[1])


# Zygote check

# params = map(x -> torch.as_tensor(copy(x)).to(device = linwrap.device, dtype = linwrap.dtype).requires_grad_(true), linwrap.params)
# torch_out = linwrap.torch_stateless_module(params, linwrap.buffers, map(z->torch.as_tensor(PyReverseDims(z)).to(dtype=linwrap.dtype), [x])...).sum()
# torchgrad = map(x-> copy(x.numpy()), torch.autograd.grad(torch_out, params))
# grad,  = Zygote.gradient(m->sum(m(x)), linwrap)
# @test length(torchgrad) == length(grad.params)
# for i in 1:length(grad.params)
#     @test isapprox(torchgrad[i], grad.params[i])
# end
# @test length(grad.params) == length(linwrap.params)
# @test grad.params[1] !== nothing
# @test grad.params[2] !== nothing
# @test size(grad.params[1]) == size(linwrap.params[1])
# @test size(grad.params[2]) == size(linwrap.params[2])

# params = map(x -> torch.as_tensor(copy(x)).to(device = linwrap.device, dtype = linwrap.dtype).requires_grad_(true), linwrap.params)
# xtorch = torch.as_tensor(PyReverseDims(copy(x))).to(dtype=linwrap.dtype).requires_grad_(true)
# torch_out = linwrap.torch_stateless_module(params, linwrap.buffers, xtorch).sum()
# torchgrad = map(x-> ReverseDimsArray(x.numpy()), torch.autograd.grad(torch_out, xtorch))[1]
# grad, = Zygote.gradient(z->sum(linwrap(z)), copy(x))
# @test size(grad) == size(x)
# @test length(torchgrad) == length(grad)
# @test isapprox(torchgrad, grad)

# Flux check
@testset "flux" begin
    lin = torch.nn.Linear(indim, outdim)
    linwrap = TorchModuleWrapper(lin)
    nn = Chain(Dense(4, 3), linwrap)
    x2 = randn(Float32, 4, batchsize)
    grad,  = Zygote.gradient(m->sum(m(x2)), nn)
    @test grad !== nothing    
end


@testset "conv" begin
    model = torch.nn.Sequential(
            torch.nn.Conv2d(1,2,5),
            torch.nn.ReLU(),
            torch.nn.Conv2d(2,6,5),
            torch.nn.ReLU()
            )
    modelwrap = TorchModuleWrapper(model)

    input = randn(Float32, 12, 12, 1, batchsize)
    #output = modelwrap(input)

    compare_grad_wrt_params(modelwrap, deepcopy(input))
    compare_grad_wrt_inputs(modelwrap, deepcopy(input))
end