module Jax

using PyCall
using ChainRulesCore
using DLPack
using Adapt
using Requires

using ..PyCallChainRules: PyAdaptor, fmap

const inspect = PyNULL()
const jax = PyNULL()
const dlpack = PyNULL()
const stax = PyNULL()
const numpy = PyNULL()

const ispysetup = Ref{Bool}(false)

pyto_dlpack(x) = @pycall dlpack.to_dlpack(x)::PyObject
pyfrom_dlpack(x) = @pycall dlpack.from_dlpack(x)::PyObject

### XXX: what's a little piracy between us
### allows empty parameter tuples
DLPack.wrap(o::Tuple{}, to_dlpack) = o
DLPack.share(o::Tuple{}, ::Type{PyObject}, from_dlpack) = o


struct JaxFunctionWrapper
    jaxfn::PyObject
end

function (wrap::JaxFunctionWrapper)(args...; kwargs...)
    out = wrap.jaxfn(fmap(x->DLPack.share(x, PyObject, pyfrom_dlpack), args)...)
    return fmap(x->DLPack.wrap(x, pyto_dlpack), out)
end

function ChainRulesCore.rrule(wrap::JaxFunctionWrapper, args...; kwargs...)
    T = typeof(first(args))
    project = ProjectTo(args)
    pyargs = fmap(x->DLPack.share(x, PyObject, pyfrom_dlpack), args)
    jax_primal, jax_vjpfun = jax.vjp(wrap.jaxfn, pyargs...; kwargs...)
    function JaxFunctionWrapper_pullback(Δ)
        cΔ = fmap(x->Adapt.adapt(PyAdaptor{T}(), x), Δ)
        dlΔ = fmap(x->DLPack.share(x, PyObject, pyfrom_dlpack), cΔ)
        tangent_vals = fmap(x->DLPack.wrap(x, pyto_dlpack), jax_vjpfun(dlΔ))
        return (NoTangent(), project(tangent_vals)...)
    end
    return fmap(x->DLPack.wrap(x, pyto_dlpack), jax_primal), JaxFunctionWrapper_pullback
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
        """
        @debug err   
        ispysetup[] = false             
        #rethrow(err)
    end
    @require Lux = "b2108857-7c20-44ae-9111-449ecde12c47" begin
        include("lux.jl")
    end
end

end