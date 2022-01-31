using PyCallChainRules: ReverseDimsArray

using PyCallChainRules.Jax: JaxFunctionWrapper, jax, numpy, stax, pyto_dlpack, ispysetup

using Test
using ChainRulesTestUtils
using Zygote
using ChainRulesCore: NoTangent
using Random
using PyCall
using DLPack
#using Flux

if !ispysetup[]
    return
end

function reversedims(a::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    permutedims(a, N:-1:1)
end

@testset "dlpack" begin
    key = jax.random.PRNGKey(0)
    for dims in ((10,), (1, 10), (2, 3, 5), (2, 3, 4, 5))
        xto = jax.random.normal(key, dims)
        xjl = DLArray(xto, pyto_dlpack).data
        @test isapprox(numpy.array(xto), xjl)
    end
end

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
#test_rrule(linwrap, params_np, x; check_inferred=false, check_thunked_output_tangent=false, rtol=1e-4, atol=1e-4)

# Zygote check
grad,  = Zygote.gradient(p->sum(linwrap(p, x)), params_np)
py"""
import jax
import jax.numpy as jnp
def grad(fn, params, x):
    f2 = lambda p, z: jnp.sum(fn(p, z))
    return jax.grad(f2)(params, x)
"""
jaxgrad = map(x->ReverseDimsArray(DLArray(x, pyto_dlpack)), (py"grad")(apply_lin, params, PyReverseDims(x)))
@test length(grad) == length(params_np)
@test size(grad[1]) == size(params_np[1])
@test size(grad[2]) == size(params_np[2])
@test isapprox(grad[1], jaxgrad[1])
@test isapprox(grad[2], jaxgrad[2])

grad, = Zygote.gradient(z->sum(linwrap(params_np, z)), x)
@test size(grad) == size(x)
py"""
import jax
import jax.numpy as jnp
def gradx(fn, params, x):
    f2 = lambda p, z: jnp.sum(fn(p, z))
    return jax.grad(f2, argnums=(1,))(params, x)
"""
jaxgrad = map(x->ReverseDimsArray(DLArray(x, pyto_dlpack)), (py"gradx")(apply_lin, params, PyReverseDims(x)))
@test isapprox(jaxgrad[1], grad)