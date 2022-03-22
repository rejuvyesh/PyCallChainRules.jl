Adapt.adapt_storage(to::PyAdaptor{T}, x::CUDA.AnyCuArray) where {T} = CUDA.CuArray(x)
Adapt.adapt_storage(to::PyAdaptor{T}, x::CUDA.StridedCuArray) where {T} = x
Adapt.adapt_storage(to::PyAdaptor{<:CUDA.CuArray}, x::FillArrays.AbstractFill) = CUDA.cu(collect(x))
Adapt.adapt_structure(to::PyAdaptor{<:CUDA.CuArray}, x::A) where {A <: SubArray}  = CUDA.cu(collect(x))