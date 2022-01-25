module Jax

using PyCall
using ChainRulesCore
using DLPack

const inspect = PyNULL()
const jax = PyNULL()
const dlpack = PyNULL()
const stax = PyNULL()
const numpy = PyNULL()

const ispysetup = Ref{Bool}(false)

function ReverseDimsArray(a::AbstractArray{T,N}) where {T<:AbstractFloat, N}
    PermutedDimsArray(a, N:-1:1)
end

rowmajor2colmajor(a::AbstractArray{T,2}) where {T<:AbstractFloat} = a
rowmajor2colmajor(a::AbstractArray{T,1}) where {T<:AbstractFloat} = a

function rowmajor2colmajor(a::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    PermutedDimsArray(reshape(a, reverse(size(a))...), N:-1:1)
end

function via_dlpack(x)
    return rowmajor2colmajor(Base.unsafe_wrap(Array, DLArray{Float32, x.ndim}(@pycall dlpack.to_dlpack(x)::PyObject)))
end


mapover(f, iselement, x) =
                  iselement(x) ? f(x) : map(e -> mapover(f, iselement, e), x)

struct JaxFunctionWrapper
    jaxfn::PyObject
end

function (wrap::JaxFunctionWrapper)(args...)
    # TODO: handle multiple outputs
    out = (wrap.jaxfn(mapover(x->jax.numpy.asarray(PyReverseDims(x)), x-> x isa Array, args)...))
    return ReverseDimsArray(via_dlpack(out))
end

function ChainRulesCore.rrule(wrap::JaxFunctionWrapper, args...)
    project = ProjectTo(args)
    jax_primal, jax_vjpfun = jax.vjp(wrap.jaxfn, mapover(x->jax.numpy.asarray(PyReverseDims(x)), x-> x isa Array, args)...)
    function JaxFunctionWrapper_pullback(Δ)
        tangent_vals = mapover(x->ReverseDimsArray(via_dlpack(x)), x-> x isa PyObject, jax_vjpfun(jax.numpy.asarray(PyReverseDims(Δ))))
        return (NoTangent(), project(tangent_vals)...)
    end
    return ReverseDimsArray(via_dlpack(jax_primal)), JaxFunctionWrapper_pullback
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