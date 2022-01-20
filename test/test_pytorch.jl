using PyCallChainRules.Torch: TorchModuleWrapper, torch, functorch

using Test
using ChainRulesTestUtils

batchsize = 1
indim = 3
outdim = 2
lin = torch.nn.Linear(indim, outdim)

linwrap = TorchModuleWrapper(lin)

x = randn(Float32, indim, batchsize)
y = linwrap(x)
@test size(y) == (outdim, batchsize)

test_rrule(linwrap, x)