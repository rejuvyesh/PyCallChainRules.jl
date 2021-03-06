using PyCallChainRules.Jax: JaxFunctionWrapper, jax, numpy, stax, pyto_dlpack, pyfrom_dlpack, ispysetup

using Test

using Zygote
using CUDA
using Random
using PyCall
using DLPack


if !ispysetup[]
    return
end

@testset "dlpack" begin
    key = jax.random.PRNGKey(0)
    for dims in ((10,), (1, 10), (2, 3, 5), (2, 3, 4, 5))
        xto = jax.random.normal(key, dims)
        xjl = DLPack.wrap(xto, pyto_dlpack)
        @test Tuple(xto.shape) == reverse(size(xjl))
        @test isapprox(sum(numpy.array(xto)), sum(xjl))
    end
end

batchsize = 1
indim = 3
outdim = 2

init_lin, apply_lin = stax.Dense(outdim)
_, params = init_lin(jax.random.PRNGKey(0), (-1, indim))
params_np = map(x->DLPack.wrap(x, pyto_dlpack), params)
linwrap = JaxFunctionWrapper(apply_lin)
x = randn(Float32, indim, batchsize)
if CUDA.functional()
    params_np = map(cu, params_np)
    x = cu(x)
end
y = linwrap(params_np, x)
@test size(y) == (outdim, batchsize)

# Zygote check
if CUDA.functional()
    params_np = map(cu, params_np)
    x = cu(x)
end

grad,  = Zygote.gradient(p->sum(linwrap(p, x)), params_np)
py"""
import jax
import jax.numpy as jnp
def grad(fn, params, x):
    f2 = lambda p, z: jnp.sum(fn(p, z))
    return jax.grad(f2)(params, x)
"""
jaxgrad = map(x->DLPack.wrap(x, pyto_dlpack), (py"grad")(apply_lin, params, DLPack.share(x, PyObject, pyfrom_dlpack)))
@test length(grad) == length(params_np)
@test size(grad[1]) == size(params_np[1])
@test size(grad[2]) == size(params_np[2])
@test isapprox(Array(grad[1]), Array(jaxgrad[1]))
@test isapprox(Array(grad[2]), Array(jaxgrad[2]))

grad, = Zygote.gradient(z->sum(linwrap(params_np, z)), x)
@test size(grad) == size(x)
py"""
import jax
import jax.numpy as jnp
def gradx(fn, params, x):
    f2 = lambda p, z: jnp.sum(fn(p, z))
    return jax.grad(f2, argnums=(1,))(params, x)
"""
jaxgrad = map(x->DLPack.wrap(x, pyto_dlpack), (py"gradx")(apply_lin, params, DLPack.share(x, PyObject, pyfrom_dlpack)))
@test isapprox(Array(jaxgrad[1]), Array(grad))