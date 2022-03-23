using DiffEqFlux
using OrdinaryDiffEq
using Optimisers
using Flux
using CUDA

CUDA.allowscalar(false)

using PyCall
using PyCallChainRules.Torch: TorchModuleWrapper, torch

u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0, 1.5f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = (Array(solve(prob,Tsit5(),saveat=t)))
ode_data = CUDA.cu(ode_data)


torch_module = torch.nn.Sequential(
    torch.nn.Linear(2, 50), torch.nn.Tanh(),
    torch.nn.Linear(50, 2)
)

torch_module = torch_module.to(device=torch.device("cuda"))


jlmod = TorchModuleWrapper(torch_module)
p, re = Optimisers.destructure(jlmod)

u0 = CUDA.cu(u0)

dudt(u, p, t) = re(p)(u)
prob = ODEProblem(dudt, u0, tspan)

function predict_n_ode_gpu(p)
    Flux.gpu(solve(prob,Tsit5(),u0=u0,p=p,saveat=t))
end
  
function loss_n_ode_gpu(p)
    pred = predict_n_ode_gpu(p)
    loss = sum(abs2,ode_data .- pred)
    loss
end

@info "before" loss_n_ode_gpu(p)

function train(p;nsteps=1000)
    opt = Optimisers.ADAM(0.01)
    state = Optimisers.setup(opt, p)

    for i in 1:nsteps
        gs, = Flux.gradient(p) do ps
            loss_n_ode_gpu(ps)
        end
        state, p = Optimisers.update(state, p, gs)
    end
    return p
end

newp = train(p)

@info "after" loss_n_ode_gpu(newp)