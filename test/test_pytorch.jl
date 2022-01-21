using PyCallChainRules.Torch: TorchModuleWrapper, torch, functorch

using Test
using ChainRulesTestUtils
#using Zygote
#using Flux
using ChainRulesCore: NoTangent, AbstractZero
import Random
using PyCall
using Infiltrator
ChainRulesTestUtils.rand_tangent(rng::Random.AbstractRNG, x::Ptr) = NoTangent()
ChainRulesTestUtils.test_approx(::AbstractZero, x::PyObject, msg=""; kwargs...) = @test true

function ChainRulesTestUtils.FiniteDifferences.to_vec(x::TorchModuleWrapper) 
    params_vec, back = ChainRulesTestUtils.FiniteDifferences.to_vec(x.params)
    function TorchModuleWrapper_from_vec(params_vec)
        TorchModuleWrapper(x.torch_stateless_module, x.dtype, x.device, back(params_vec))
    end
    return params_vec, TorchModuleWrapper_from_vec
end

batchsize = 1
indim = 3
outdim = 2
lin = torch.nn.Linear(indim, outdim)

linwrap = TorchModuleWrapper(lin)

x = randn(Float32, indim, batchsize)
y = linwrap(x)
@test size(y) == (outdim, batchsize)

# CRTU check TODO
x = randn(Float32, indim, batchsize)
test_rrule(linwrap, x; check_inferred=false, check_thunked_output_tangent=false, atol=1e-4, rtol=1e-4)
# const CRTU = ChainRulesTestUtils
# primals_and_tangents = CRTU.auto_primal_and_tangent((linwrap, x))
# CRTU.primal(primals_and_tangents)
# CRTU.tangent(primals_and_tangents)
# primals = CRTU.primal(primals_and_tangents)
# accum_cotangents = CRTU.tangent(primals_and_tangents)
# using ChainRulesCore: rrule
# config = CRTU.ADviaRuleConfig()
# res = rrule(config, primals...)
# y_ad, pullback = res
# call(f, xs...) = f(xs...;)
# call(primals...)
# y = call(primals...)
# ȳ = CRTU.rand_tangent(y)
# ad_cotangents = pullback(ȳ)
# length(primals)
# length(ad_cotangents)
# ad_cotangents[1]
# ad_cotangents[2]
# is_ignored = isa.(accum_cotangents, NoTangent)
# fd_cotangents =  CRTU._make_j′vp_call(CRTU._fdm, call, ȳ, primals, is_ignored)
#call2 = CRTU._wrap_function(call, primals, is_ignored)
#CRTU.test_approx(ad_cotangents[1], fd_cotangents[1])



# # Zygote check
# grad,  = Zygote.gradient(m->sum(m(x)), linwrap)
# @test length(grad.params) == 2
# @test grad.params[1] !== nothing
# @test grad.params[2] !== nothing
# @test size(grad.params[1]) == size(linwrap.params[1])
# @test size(grad.params[2]) == size(linwrap.params[2])

# grad, = Zygote.gradient(z->sum(linwrap(z)), x)
# @test size(grad) == size(x)

# # Flux check
# nn = Chain(Dense(4, 3), linwrap)
# x2 = randn(Float32, 4, batchsize)
# grad,  = Zygote.gradient(m->sum(m(x2)), nn)


