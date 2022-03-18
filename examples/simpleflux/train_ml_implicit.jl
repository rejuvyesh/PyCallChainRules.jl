using Flux
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

opt = Flux.ADAM(0.1)

input = randn(Float32, input_dim, batchsize)
target = randn(Float32, output_dim, batchsize)

loss(x, y) = Flux.Losses.mse(jlmodel(x), y)
ps = Flux.params(jlmodel)
@info "before" map(sum, ps)
for i in 1:1
    gs = Flux.gradient(ps) do 
        loss(input, target)
    end
    @show gs.grads
    Flux.Optimise.update!(opt, ps, gs)
end
@info "after" map(sum, ps)
