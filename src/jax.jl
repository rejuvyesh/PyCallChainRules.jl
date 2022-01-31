module Jax

using PyCall
using ChainRulesCore
using DLPack

using ..PyCallChainRules: ReverseDimsArray

const inspect = PyNULL()
const jax = PyNULL()
const dlpack = PyNULL()
const stax = PyNULL()
const numpy = PyNULL()

const ispysetup = Ref{Bool}(false)

pyto_dlpack(x) = @pycall dlpack.to_dlpack(x)::PyObject

mapover(f, iselement, x) =
                  iselement(x) ? f(x) : map(e -> mapover(f, iselement, e), x)

struct JaxFunctionWrapper
    jaxfn::PyObject
end

function (wrap::JaxFunctionWrapper)(args...)
    # TODO: handle multiple outputs
    out = (wrap.jaxfn(mapover(x->jax.numpy.asarray(PyReverseDims(x)), x-> x isa Array, args)...))
    return ReverseDimsArray(DLArray(out, pyto_dlpack))
end

function ChainRulesCore.rrule(wrap::JaxFunctionWrapper, args...)
    project = ProjectTo(args)
    jax_primal, jax_vjpfun = jax.vjp(wrap.jaxfn, mapover(x->jax.numpy.asarray(PyReverseDims(x)), x-> x isa Array, args)...)
    function JaxFunctionWrapper_pullback(Δ)
        tangent_vals = mapover(x->ReverseDimsArray(DLArray(x, pyto_dlpack)), x-> x isa PyObject, jax_vjpfun(jax.numpy.asarray(PyReverseDims(Δ))))
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