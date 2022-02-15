module PyCallChainRules

using DLPack
using Functors
using FillArrays
using CUDA


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
