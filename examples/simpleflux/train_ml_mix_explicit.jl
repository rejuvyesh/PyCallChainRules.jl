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

# Mix of Flux layers and Torch layers
jlmodel = Chain(Dense(input_dim, input_dim, Flux.relu), 
                TorchModuleWrapper(torch_module), x->Flux.relu.(x), 
                Dense(output_dim, output_dim))


input = randn(Float32, input_dim, batchsize)
target = randn(Float32, output_dim, batchsize)

loss(model, x, y) = Flux.Losses.mse(model(x), y)

@info "before" loss(jlmodel, input, target)

function train(model;nsteps=100)
    opt = Optimisers.ADAM(0.01)
    state = Optimisers.setup(opt, model)
    
    for i in 1:nsteps
        gs, _ = Flux.gradient(model, input, target) do m, x, y
            loss(m, x, y)
        end
        state, model = Optimisers.update(state, model, gs)
    end
    return model
end

jlmodel = train(jlmodel)

@info "after" loss(jlmodel, input, target)
