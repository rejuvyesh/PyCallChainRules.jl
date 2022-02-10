module PyCallChainRules

using DLPack
using Adapt
using Functors
using FillArrays
using CUDA

function ReverseDimsArray(a::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    PermutedDimsArray(a, N:-1:1)
end

maybecontiguous(x::AbstractArray) = Array(x)
mayebecontiguous(x::StridedArray) = x
function maybecontiguous(x::FillArrays.AbstractFill) 
    x = collect(x)
    if CUDA.functional()
        x = CUDA.cu(x)
    end
    return x
end
maybecontiguous(x::AnyCuArray) = CuArray(x)
maybecontiguous(x::StridedCuArray) = x

# Write your package code here.
include("pytorch.jl")

include("jax.jl")

end
