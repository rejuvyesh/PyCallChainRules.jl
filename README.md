# PyCallChainRules

While Julia is great, there are still a lot of existing useful differentiable python code in PyTorch, Jax, etc. Given PyCall.jl is already so great and seamless, one might wonder what it takes to differentiate through those `pycall`s. This library aims for that ideal.

## Basic Usage

### PyTorch

```julia
using PyCallChainRules.Torch: TorchModuleWrapper, torch
using Zygote

indim = 32
outdim = 16
torch_module = torch.nn.Linear(indim, outdim) # Can be anything subclassing torch.nn.Module
jlwrap = TorchModuleWrapper(torch_module)

batchsize = 64
input = randn(Float32, indim, batchsize)
output = jlwrap(input)

target = randn(Float32, outdim, batchsize)
loss(m, x, y) = sum(m(x) .- target)
grad, = Zygote.gradient(m->loss(m, input, target), jlwrap)
```

### Jax

## Current Limitations / TODO

- CPU only
- Lots of array copies
- Assumes wrapped python functions are single output only