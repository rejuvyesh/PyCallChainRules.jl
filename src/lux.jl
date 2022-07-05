import .Lux

using Random
using DLPack
using PyCall
using Functors

using PyCallChainRules.Jax: JaxFunctionWrapper, jax, pyto_dlpack

struct LuxStaxWrapper{N} <: Lux.AbstractExplicitLayer
    initfn::PyObject
    applyfn::JaxFunctionWrapper
    input_shape::NTuple{N, Int}
end

function LuxStaxWrapper(init::PyObject, apply::PyObject; input_shape::NTuple{N,Int}) where {N}
    apply_jl = JaxFunctionWrapper(apply)
    return LuxStaxWrapper{N}(init, apply_jl, input_shape)
end

function Lux.initialparameters(rng::AbstractRNG, l::LuxStaxWrapper)
    val = abs(rand(rng, Int32))
    _, params = l.initfn(jax.random.PRNGKey(val), l.input_shape)
    params_jl = fmap(x->DLPack.wrap(x, pyto_dlpack), params)
    return params_jl
end

function (model::LuxStaxWrapper)(x, ps, st::NamedTuple)
    model.applyfn(ps, x), st
end