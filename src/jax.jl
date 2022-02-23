module Jax

using PyCall
using ChainRulesCore
using DLPack
using FillArrays: AbstractFill
using Functors: fmap
using Adapt

using ..PyCallChainRules: PyAdaptor

const inspect = PyNULL()
const jax = PyNULL()
const dlpack = PyNULL()
const stax = PyNULL()
const numpy = PyNULL()

const ispysetup = Ref{Bool}(false)

pyto_dlpack(x) = @pycall dlpack.to_dlpack(x)::PyObject
pyfrom_dlpack(x) = @pycall dlpack.from_dlpack(x)::PyObject

struct JaxFunctionWrapper
    jaxfn::PyObject
end

function (wrap::JaxFunctionWrapper)(args...; kwargs...)
    # TODO: handle multiple outputs
    out = (wrap.jaxfn(fmap(x->DLPack.share(x, PyObject, pyfrom_dlpack), args)...))
    return (DLPack.wrap(out, pyto_dlpack))
end

function ChainRulesCore.rrule(wrap::JaxFunctionWrapper, args...; kwargs...)
    T = typeof(first(args))
    project = ProjectTo(args)
    jax_primal, jax_vjpfun = jax.vjp(wrap.jaxfn, fmap(x->DLPack.share(x, PyObject, pyfrom_dlpack), args)...; kwargs...)
    function JaxFunctionWrapper_pullback(Δ)
        cΔ = Adapt.adapt(PyAdaptor{T}, Δ)
        dlΔ = DLPack.share(cΔ, PyObject, pyfrom_dlpack)
        tangent_vals = fmap(x->(DLPack.wrap(x, pyto_dlpack)), jax_vjpfun(dlΔ))
        return (NoTangent(), project(tangent_vals)...)
    end
    JaxFunctionWrapper_pullback(Δ::AbstractFill) = JaxFunctionWrapper_pullback(collect(Δ))
    return (DLPack.wrap(jax_primal, pyto_dlpack)), JaxFunctionWrapper_pullback
end


function __init__()
    try
        copy!(jax, pyimport("jax"))
        copy!(dlpack, pyimport("jax.dlpack"))
        copy!(numpy, pyimport("numpy"))
        copy!(stax, pyimport("jax.example_libraries.stax"))
        copy!(inspect, pyimport("inspect"))
        ispysetup[] = true
    catch err
        @warn """PyCallChainRules.jl has failed to import jax from Python.
                 Please make sure these are installed. 
                 methods of this package.
        """
        @debug err   
        ispysetup[] = false             
        #rethrow(err)
    end
end

end