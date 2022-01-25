# PyCallChainRules

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://rejuvyesh.github.io/PyCallChainRules.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://rejuvyesh.github.io/PyCallChainRules.jl/dev)

While Julia is great, there are still a lot of existing useful differentiable python code in PyTorch, Jax, etc. Given PyCall.jl is already so great and seamless, one might wonder what it takes to differentiate through those `pycall`s. This library aims for that ideal.

## Basic Usage


### PyTorch

**Install Python dependencies**:
```julia
using PyCall
run(`$(PyCall.pyprogramname) -m pip install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html --upgrade`)
run(`$(PyCall.pyprgramname) -m pip install "git+https://github.com/pytorch/functorch.git"`)
```

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

**Install Python dependencies**:
```julia
using PyCall
run(`$(PyCall.pyprogramname) -m pip install jax\["cpu"\])
```

## Current Limitations / TODO

- CPU only
- Lots of array copies
- Assumes wrapped python functions are single output only