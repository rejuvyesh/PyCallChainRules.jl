using PyCallChainRules.Torch: TorchModuleWrapper, torch, functorch, ispysetup, ReverseDimsArray

using Test
using ChainRulesTestUtils
using Zygote
using Flux
using ChainRulesCore: NoTangent, AbstractZero
import Random
using PyCall

if !ispysetup[]
    return
end

Random.seed!(42)
ChainRulesTestUtils.rand_tangent(rng::Random.AbstractRNG, x::Ptr) = NoTangent()
ChainRulesTestUtils.test_approx(::AbstractZero, x::PyObject, msg=""; kwargs...) = @test true
ChainRulesTestUtils.test_approx(::AbstractZero, x::Tuple{}, msg=""; kwargs...) = @test true

function ChainRulesTestUtils.FiniteDifferences.to_vec(x::TorchModuleWrapper) 
    params_vec, back = ChainRulesTestUtils.FiniteDifferences.to_vec(x.params)
    function TorchModuleWrapper_from_vec(params_vec)
        TorchModuleWrapper(x.torch_stateless_module, x.dtype, x.device, back(params_vec), x.buffers)
    end
    return params_vec, TorchModuleWrapper_from_vec
end

batchsize = 1
indim = 3
outdim = 2
hiddendim = 4
lin = torch.nn.Sequential(torch.nn.Linear(indim, hiddendim), torch.nn.ReLU(), torch.nn.Linear(hiddendim, outdim))

linwrap = TorchModuleWrapper(lin)

x = randn(Float32, indim, batchsize)
y = linwrap(x)
@test size(y) == (outdim, batchsize)

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
grad,  = Zygote.gradient(m->sum(m(x)), linwrap)
params = map(x -> torch.as_tensor(x).to(device = linwrap.device, dtype = linwrap.dtype).requires_grad_(true), linwrap.params)
torch_out = linwrap.torch_stateless_module(params, linwrap.buffers, map(z->torch.as_tensor(PyReverseDims(z)).to(dtype=linwrap.dtype), [x])...).sum()
torchgrad = map(x-> x.numpy(), torch.autograd.grad(torch_out, params))
@test length(torchgrad) == length(grad.params)
for i in 1:length(grad.params)
    @test isapprox(torchgrad[i], grad.params[i])
end
@test length(grad.params) == length(linwrap.params)
@test grad.params[1] !== nothing
@test grad.params[2] !== nothing
@test size(grad.params[1]) == size(linwrap.params[1])
@test size(grad.params[2]) == size(linwrap.params[2])

grad, = Zygote.gradient(z->sum(linwrap(z)), x)
@test size(grad) == size(x)
params = map(x -> torch.as_tensor(x).to(device = linwrap.device, dtype = linwrap.dtype).requires_grad_(true), linwrap.params)
xtorch = torch.as_tensor(PyReverseDims(x)).to(dtype=linwrap.dtype).requires_grad_(true)
torch_out = linwrap.torch_stateless_module(params, linwrap.buffers, xtorch).sum()
torchgrad = map(x-> ReverseDimsArray(x.numpy()), torch.autograd.grad(torch_out, xtorch))[1]
@test length(torchgrad) == length(grad)
@test isapprox(torchgrad, grad)

# Flux check
nn = Chain(Dense(4, 3), linwrap)
x2 = randn(Float32, 4, batchsize)
grad,  = Zygote.gradient(m->sum(m(x2)), nn)


model = torch.nn.Sequential(
          torch.nn.Conv2d(1,2,5),
          torch.nn.ReLU(),
          torch.nn.Conv2d(2,6,5),
          torch.nn.ReLU()
        )
modelwrap = TorchModuleWrapper(model)

input = randn(Float32, 12, 12, 1, batchsize)
#output = modelwrap(input)

x = input
grad,  = Zygote.gradient(m->sum(m(x)), modelwrap)
params = map(x -> torch.as_tensor(x).to(device = modelwrap.device, dtype = modelwrap.dtype).requires_grad_(true), modelwrap.params)
torch_out = modelwrap.torch_stateless_module(params, modelwrap.buffers, map(z->torch.as_tensor(PyReverseDims(z)).to(dtype=modelwrap.dtype), [x])...).sum()
torchgrad = map(x-> x.numpy(), torch.autograd.grad(torch_out, params))
@test length(torchgrad) == length(grad.params)
for i in 1:length(grad.params)
    @test isapprox(torchgrad[i], grad.params[i])
end
@test length(grad.params) == length(modelwrap.params)
@test grad.params[1] !== nothing
@test grad.params[2] !== nothing
@test size(grad.params[1]) == size(modelwrap.params[1])
@test size(grad.params[2]) == size(modelwrap.params[2])

grad, = Zygote.gradient(z->sum(modelwrap(z)), x)
@test size(grad) == size(x)
params = map(x -> torch.as_tensor(x).to(device = modelwrap.device, dtype = modelwrap.dtype).requires_grad_(true), modelwrap.params)
xtorch = torch.as_tensor(PyReverseDims(x)).to(dtype=modelwrap.dtype).requires_grad_(true)
torch_out = modelwrap.torch_stateless_module(params, modelwrap.buffers, xtorch).sum()
torchgrad = map(x-> ReverseDimsArray(x.numpy()), torch.autograd.grad(torch_out, xtorch))[1]
@test length(torchgrad) == length(grad)
@test isapprox(torchgrad, grad)

#test_rrule(modelwrap, input; check_inferred=false, check_thunked_output_tangent=false, atol=1e-2, rtol=1e-2)