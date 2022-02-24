module PyCallChainRules

using DLPack
using Requires

import FillArrays
import Adapt
import Functors: fmap
import ChainRulesCore

struct PyAdaptor{T} end
Adapt.adapt_storage(to::PyAdaptor{T}, x::AbstractArray) where {T} = convert(Array, x)
Adapt.adapt_storage(to::PyAdaptor{T}, x::StridedArray) where {T} = x
Adapt.adapt_storage(to::PyAdaptor{<:AbstractArray}, x::FillArrays.AbstractFill) = collect(x)

### XXX: what's a little piracy between us
fmap(f, x::ChainRulesCore.Tangent) = fmap(f, x.backing)


# Write your package code here.
include("pytorch.jl")

include("jax.jl")

function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("cuda.jl")
    end    
end

end
