module PyCallChainRules

using PyCall: PyObject, @pycall, f_contiguous
using DLPack

function ReverseDimsArray(a::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    PermutedDimsArray(a, N:-1:1)
end

rowmajor2colmajor(a::AbstractArray{T,2}) where {T<:AbstractFloat} = a
rowmajor2colmajor(a::AbstractArray{T,1}) where {T<:AbstractFloat} = a

function rowmajor2colmajor(a::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    PermutedDimsArray(reshape(a, reverse(size(a))...), N:-1:1)
end

function via_dlpack(dlpack, x)
    dla = DLArray(@pycall dlpack.to_dlpack(x)::PyObject)
    if DLPack.device_type(dla) == DLPack.kDLCPU
        arr = Base.unsafe_wrap(Array, dla)
    # elseif DLPack.device_type(dla) == DLPack.kDLCUDA
    #     arr = Base.unsafe_wrap(CuArray, dla)
    end
    sz = size(dla)
    st = strides(dla)
    T = eltype(dla)
    if !f_contiguous(T, sz, st)
        arr = rowmajor2colmajor(arr)
    end
    return arr
end

# Write your package code here.
include("pytorch.jl")

include("jax.jl")

end
