using PyCallChainRules.Torch: TorchModuleWrapper, torch, functorch

using Test
using ChainRulesTestUtils
using Zygote

batchsize = 1
indim = 3
outdim = 2
lin = torch.nn.Linear(indim, outdim)

linwrap = TorchModuleWrapper(lin)

x = randn(Float32, indim, batchsize)
y = linwrap(x)
@test size(y) == (outdim, batchsize)

# Zygote check
grad,  = Zygote.gradient(m->sum(m(x)), linwrap)
@test length(grad.params) == 2
@test grad.params[1] !== nothing
@test grad.params[2] !== nothing
@test size(grad.params[1]) == size(linwrap.params[1])
@test size(grad.params[2]) == size(linwrap.params[2])

grad, = Zygote.gradient(z->sum(linwrap(z)), x)
@test size(grad) == size(x)

#test_rrule(linwrap, x)
