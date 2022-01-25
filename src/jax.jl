module Jax

using PythonCall
using PythonCall: pynew, pycopy!
using ChainRulesCore

const inspect = pynew()
const jax = pynew()
const stax = pynew()
const numpy = pynew()

const ispysetup = Ref{Bool}(false)

function reversedims(a::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    permutedims(a, N:-1:1)
end

mapover(f, iselement, x) =
                  iselement(x) ? f(x) : map(e -> mapover(f, iselement, e), x)

struct JaxFunctionWrapper
    jaxfn::Py
end

function (wrap::JaxFunctionWrapper)(args...)
    # TODO: handle multiple outputs
    out = numpy.array(wrap.jaxfn(mapover(x->jax.numpy.asarray(PyReverseDims(x)), x-> x isa Array, args)...))
    return reversedims((out))
end

function ChainRulesCore.rrule(wrap::JaxFunctionWrapper, args...)
    project = ProjectTo(args)
    jax_primal, jax_vjpfun = jax.vjp(wrap.jaxfn, mapover(x->jax.numpy.asarray(PyReverseDims(x)), x-> x isa Array, args)...)
    function JaxFunctionWrapper_pullback(Δ)
        tangent_vals = mapover(x->reversedims(numpy.array(x)), x-> x isa Py,jax_vjpfun(jax.numpy.array(PyReverseDims(Δ))))

        return (NoTangent(), project(tangent_vals)...)
    end
    return reversedims(numpy.array(jax_primal)), JaxFunctionWrapper_pullback
end


function __init__()
    try
        pycopy!(jax, pyimport("jax"))
        pycopy!(numpy, pyimport("numpy"))
        pycopy!(stax, pyimport("jax.example_libraries.stax"))
        pycopy!(inspect, pyimport("inspect"))
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