using SafeTestsets

@safetestset "PyCallChainRules.jl" begin
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
