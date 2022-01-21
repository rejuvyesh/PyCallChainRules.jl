using PyCallChainRules.Torch: TorchModuleWrapper, torch, functorch

using Test
using ChainRulesTestUtils
using Zygote
using Flux
using ChainRulesCore: NoTangent
import Random
using PyCall
ChainRulesTestUtils.rand_tangent(rng::Random.AbstractRNG, x::Ptr) = NoTangent()
ChainRulesTestUtils.FiniteDifferences.to_vec(x::Ptr{PyCall.PyObject_struct}) = (Bool[], _ -> x)

batchsize = 1
indim = 3
outdim = 2
lin = torch.nn.Linear(indim, outdim)

linwrap = TorchModuleWrapper(lin)

x = randn(Float32, indim, batchsize)
y = linwrap(x)
@test size(y) == (outdim, batchsize)

# CRTU check TODO
x = randn(Float32, indim, batchsize)
# test_rrule(linwrap, x; check_inferred=false)

# Zygote check
grad,  = Zygote.gradient(m->sum(m(x)), linwrap)
@test length(grad.params) == 2
@test grad.params[1] !== nothing
@test grad.params[2] !== nothing
@test size(grad.params[1]) == size(linwrap.params[1])
@test size(grad.params[2]) == size(linwrap.params[2])

grad, = Zygote.gradient(z->sum(linwrap(z)), x)
@test size(grad) == size(x)

# Flux check
nn = Chain(Dense(4, 3), linwrap)
x2 = randn(Float32, 4, batchsize)
grad,  = Zygote.gradient(m->sum(m(x2)), nn)


