# PyCallChainRules

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://rejuvyesh.github.io/PyCallChainRules.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://rejuvyesh.github.io/PyCallChainRules.jl/dev)

While Julia is great, there are still a lot of existing useful differentiable python code in PyTorch, Jax, etc. Given PyCall.jl is already so great and seamless, one might wonder what it takes to differentiate through those `pycall`s. This library aims for that ideal.

Thanks to [@pabloferz](https://github.cim/pabloferz), this works on both CPU and GPU without any array copies via [DLPack.jl](https://github.com/pabloferz/DLPack.jl).

## Basic Usage


### PyTorch

#### CPU only

##### Install Python dependencies

```julia
using PyCall
run(`$(PyCall.pyprogramname) -m pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html`)
run(`$(PyCall.pyprogramname) -m pip install functorch`)
```

##### Example

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

#### GPU

##### Install Python dependencies

```julia
using PyCall
# For CUDA 11 and PyTorch 1.11
run(`$(PyCall.pyprogramname) -m pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`)
run(`$(PyCall.pyprogramname) -m pip install functorch`)
```

##### Example

```julia
using CUDA
using PyCallChainRules.Torch: TorchModuleWrapper, torch
using Zygote

@assert CUDA.isfunctional()

indim = 32
outdim = 16
torch_module = torch.nn.Linear(indim, outdim).to(device=torch.device("cuda:0")) # Can be anything subclassing torch.nn.Module
jlwrap = TorchModuleWrapper(torch_module)

batchsize = 64
input = CUDA.cu(randn(Float32, indim, batchsize))
output = jlwrap(input)

target = CUDA.cu(randn(Float32, outdim, batchsize))
loss(m, x, y) = sum(m(x) .- y)
grad, = Zygote.gradient(m->loss(m, input, target), jlwrap)
```


### Jax

#### CPU only 

##### Install Python dependencies
```julia
using PyCall
run(`$(PyCall.pyprogramname) -m pip install jax\["cpu"\]`) # for cpu version
```

##### Example
```julia
using PyCallChainRules.Jax: JaxFunctionWrapper, jax, stax, pyto_dlpack

batchsize = 64
indim = 32
outdim = 16

init_lin, apply_lin = stax.Dense(outdim)
_, params = init_lin(jax.random.PRNGKey(0), (-1, indim))
params_jl = map(x->DLPack.wrap(x, pyto_dlpack), params)
jlwrap = JaxFunctionWrapper(jax.jit(apply_lin))
input = randn(Float32, indim, batchsize)
output = jlwrap(params_jl, input)

target = randn(Float32, outdim, batchsize)
loss(p, x, y) = sum(jlwrap(p, x) .- y)
grad, = Zygote.gradient(p->loss(p, input, target), params_jl)
```

#### GPU

##### Install Python dependencies
```julia
using PyCall
run(`$(PyCall.pyprogramname) -m pip install jax\["cuda"\] -f https://storage.googleapis.com/jax-releases/jax_releases.html`)
```

##### Example
```julia
using PyCallChainRules.Jax: JaxFunctionWrapper, jax, stax
using CUDA

using PyCallChainRules.Jax: JaxFunctionWrapper, jax, stax, pyto_dlpack

batchsize = 64
indim = 32
outdim = 16

init_lin, apply_lin = stax.Dense(outdim)
_, params = init_lin(jax.random.PRNGKey(0), (-1, indim))
params_jl = map(x->DLPack.wrap(x, pyto_dlpack), params)
jlwrap = JaxFunctionWrapper(jax.jit(apply_lin))
input = CUDA.cu(randn(Float32, indim, batchsize))
output = jlwrap(params_jl, input)

target = CUDA.cu(randn(Float32, outdim, batchsize))
loss(p, x, y) = sum(jlwrap(p, x) .- y)
grad, = Zygote.gradient(p->loss(p, input, target), params_jl)
```


## Current Limitations

- Input and output types of wrapped python functions can only be python tensors or [nested] tuples of python tensors.
- Keyword arguments should not be arrays and do not support differentiation.
