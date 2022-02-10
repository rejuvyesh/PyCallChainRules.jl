module Jax

using PyCall
using ChainRulesCore
using DLPack
using Functors: fmap

using ..PyCallChainRules: ReverseDimsArray, maybecontiguous

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

function (wrap::JaxFunctionWrapper)(args...)
    # TODO: handle multiple outputs
    out = (wrap.jaxfn(fmap(x->DLPack.share(x, pyfrom_dlpack), args)...))
    return ReverseDimsArray(DLArray(out, pyto_dlpack))
end

function ChainRulesCore.rrule(wrap::JaxFunctionWrapper, args...)
    project = ProjectTo(args)
    jax_primal, jax_vjpfun = jax.vjp(wrap.jaxfn, fmap(x->DLPack.share(x, pyfrom_dlpack), args)...)
    function JaxFunctionWrapper_pullback(Δ)
        cΔ = maybecontiguous(Δ)
        dlΔ = DLPack.share(cΔ, pyfrom_dlpack)
        tangent_vals = mapover(x->ReverseDimsArray(DLArray(x, pyto_dlpack)), x-> x isa PyObject, jax_vjpfun(dlΔ))
        return (NoTangent(), project(tangent_vals)...)
    end
    return ReverseDimsArray(DLArray(jax_primal, pyto_dlpack)), JaxFunctionWrapper_pullback
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