using PyCallChainRules.Jax: JaxFunctionWrapper, jax, numpy, stax, reversedims

using Test
using ChainRulesTestUtils
using Zygote
using ChainRulesCore: NoTangent
using Random
#using Flux

### !!! Remove this piracy once fixed on CRTU
ChainRulesTestUtils.rand_tangent(rng::Random.AbstractRNG, x::Ptr) = NoTangent()

batchsize = 1
indim = 3
outdim = 2

init_lin, apply_lin = stax.Dense(outdim)
_, params = init_lin(jax.random.PRNGKey(0), (-1, indim))
params_np = map(reversedims ∘ numpy.array, params)
linwrap = JaxFunctionWrapper(apply_lin)
x = randn(Float32, indim, batchsize)
y = linwrap(params_np, x)
@test size(y) == (outdim, batchsize)

# CRTU check TODO
test_rrule(linwrap, params_np, x; check_inferred=false, check_thunked_output_tangent=false, rtol=1e-4, atol=1e-4)

# Zygote check
grad,  = Zygote.gradient(p->sum(linwrap(p, x)), params_np)
@test length(grad) == length(params_np)
@test size(grad[1]) == size(params_np[1])
@test size(grad[2]) == size(params_np[2])

grad, = Zygote.gradient(z->sum(linwrap(params_np, z)), x)
@test size(grad) == size(x)