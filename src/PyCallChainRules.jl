module PyCallChainRules

using DLPack

function ReverseDimsArray(a::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    PermutedDimsArray(a, N:-1:1)
end

maybecontiguous(x::AbstractArray) = Array(x)
mayebecontiguous(x::StridedArray) = x
# maybecontiguous(x::AnyCuArray) = CuArray(x)
# maybecontiguous(x::StridedCuArray) = x

# Write your package code here.
include("pytorch.jl")

include("jax.jl")

end
