module Torch

using PythonCall
using PythonCall: pynew, pycopy!

using ChainRulesCore

const inspect = pynew()
const numpy = pynew()
const torch = pynew()
const functorch = pynew()

const ispysetup = Ref{Bool}(false)

function reversedims(a::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    permutedims(a, N:-1:1)
end

function ReverseDimsArray(a::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    PermutedDimsArray(a, N:-1:1)
end

struct TorchModuleWrapper
    torch_stateless_module::Py
    dtype::Py
    device::Py
    params::Vector
    buffers::Py
end


Base.show(io::IO, f::TorchModuleWrapper) = print(io, f.torch_stateless_module, " ", f.dtype, " ", f.device, "params size=", size.(f.params))

Base.length(f::TorchModuleWrapper) = length(f.params)
Base.iterate(f::TorchModuleWrapper) = iterate(f.params)
Base.iterate(f::TorchModuleWrapper, state) = iterate(f.params, state)

function TorchModuleWrapper(torch_module, device)
    pyisinstance(torch_module, torch.nn.Module) || error("Not a torch.nn.Module")
    torch_module = torch_module.to(device)
    funmod, params, buffers = functorch.make_functional_with_buffers(torch_module)
    dtype = params[1].dtype
    jlparams = map(x -> PyArray(x.detach().numpy(); copy=false), params)
    return TorchModuleWrapper(funmod, dtype, device, jlparams, buffers)
end

function TorchModuleWrapper(torch_module)
    #device = pyconvert(Bool, torch.cuda.is_available()) ? torch.device("cuda:0") : torch.device("cpu")
    device = torch.device("cpu")
    TorchModuleWrapper(torch_module, device)
end

function (wrap::TorchModuleWrapper)(args...)
    # TODO: handle multiple outputs
    tensor_out = wrap.torch_stateless_module(Tuple(map(x -> torch.as_tensor(numpy.array(x, copy=true)).to(device = wrap.device, dtype = wrap.dtype).requires_grad_(true), wrap.params)),
        wrap.buffers, map(x -> torch.as_tensor(numpy.asarray(reversedims(x))).to(dtype = wrap.dtype, device = wrap.device), args)...)
    return reversedims(PyArray(tensor_out.detach().numpy(), copy=true))
end

function ChainRulesCore.rrule(wrap::TorchModuleWrapper, args...)
    torch_primal, torch_vjpfun = functorch.vjp(wrap.torch_stateless_module, Tuple(map(x -> torch.as_tensor(numpy.array(x, copy=true)).to(device = wrap.device, dtype = wrap.dtype).requires_grad_(true), wrap.params)),
        wrap.buffers, map(x -> torch.as_tensor(numpy.asarray(reversedims(x))).to(dtype = wrap.dtype, device = wrap.device).requires_grad_(true), args)...)
    project = ProjectTo(args)
    function TorchModuleWrapper_pullback(Δ)
        torch_tangent_vals = Tuple(torch_vjpfun(torch.as_tensor(numpy.asarray(PyArray(reversedims(Δ), copy=true))).to(dtype = wrap.dtype, device = wrap.device)))
        jlparams_tangents = map(x -> PyArray(x.detach().numpy(); copy=true), torch_tangent_vals[1])
        args_tangents = project(map(x -> reversedims(PyArray(x.detach().numpy(); copy=true)), torch_tangent_vals[3:end]))
        return (Tangent{TorchModuleWrapper}(; torch_stateless_module = NoTangent(), dtype = NoTangent(), device = NoTangent(), params = jlparams_tangents, buffers = NoTangent()), args_tangents...)
    end
    return reversedims(PyArray(torch_primal.detach().numpy(); copy=true)), TorchModuleWrapper_pullback
end


function __init__()
    try
        pycopy!(torch, pyimport("torch"))
        pycopy!(numpy, pyimport("numpy"))
        pycopy!(functorch, pyimport("functorch"))
        pycopy!(inspect, pyimport("inspect"))
        ispysetup[] = true
    catch err
        @warn """PyCallChainRules.jl has failed to import torch and functorch from Python.
                 Please make sure these are installed. 
                 methods of this package.
        """
        @debug err
        ispysetup[] = false
        #rethrow(err)        
    end
end

end