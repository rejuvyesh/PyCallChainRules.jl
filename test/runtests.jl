using SafeTestsets

@safetestset "PyCallChainRules.jl" begin
    @safetestset "torch" begin
        include("test_pytorch.jl")
    end
end
