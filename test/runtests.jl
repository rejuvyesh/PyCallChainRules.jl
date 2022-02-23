using SafeTestsets

@safetestset "PyCallChainRules.jl" begin
    @testset "adapt" begin
        using Adapt
        using PyCallChainRules
        using FillArrays
        using CUDA
        @test Adapt.adapt(PyCallChainRules.PyAdaptor{Matrix{Float32}}(), Fill(1, 2, 5)) isa Array
        if CUDA.functional()
            @test Adapt.adapt(PyCallChainRules.PyAdaptor{CuMatrix{Float32}}(), Fill(1, 2, 5)) isa CuArray
        end
    end
    @safetestset "torch" begin
        include("test_pytorch.jl")
        @safetestset "hub" begin
            include("test_pytorch_hub.jl")
        end
    end
    @safetestset "jax" begin
        include("test_jax.jl")
    end
end
