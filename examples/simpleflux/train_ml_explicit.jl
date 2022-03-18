using Flux
using Optimisers
using PyCallChainRules.Torch: TorchModuleWrapper, torch

input_dim = 4
output_dim = 2
hiddendim = 16

batchsize = 8

torch_module = torch.nn.Sequential(
                            torch.nn.Linear(input_dim, hiddendim), torch.nn.ReLU(),
                            torch.nn.Linear(hiddendim, hiddendim), torch.nn.ReLU(),
                            torch.nn.Linear(hiddendim, output_dim)
                        )
jlmodel = TorchModuleWrapper(torch_module)

opt = Optimisers.ADAM(0.1)
state = Optimisers.setup(opt, jlmodel)

input = randn(Float32, input_dim, batchsize)
target = randn(Float32, output_dim, batchsize)

loss(model, x, y) = Flux.Losses.mse(model(x), y)

@info "before" map(sum, jlmodel.params)

gs, _ = Flux.gradient(jlmodel, input, target) do m, x, y
    loss(m, x, y)
end
state, jlmodel = Optimisers.update(state, jlmodel, gs)

@info "after" map(sum, jlmodel.params)
