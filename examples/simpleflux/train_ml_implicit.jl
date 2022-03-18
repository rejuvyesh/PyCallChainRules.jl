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

input = randn(Float32, input_dim, batchsize)
target = randn(Float32, output_dim, batchsize)

loss(x, y) = Flux.Losses.mse(jlmodel(x), y)
@info "before" loss(input, target)

function train(model; nsteps=100)
    opt = Flux.ADAM(0.01)
    ps = Flux.Zygote.Params(Flux.params(model))
    for i in 1:nsteps
        gs = Flux.Zygote.gradient(ps) do 
            loss(input, target)
        end
        Flux.Optimise.update!(opt, ps, gs)
    end
    return model
end

jlmodel = train(jlmodel)

@info "after" loss(input, target)
