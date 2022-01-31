module Torch

using PyCall

using ChainRulesCore
using DLPack

using ..PyCallChainRules: ReverseDimsArray

const inspect = PyNULL()
const torch = PyNULL()
const functorch = PyNULL()
const dlpack = PyNULL()
const ispysetup = Ref{Bool}(false)

pyto_dlpack(x) = @pycall dlpack.to_dlpack(x)::PyObject

struct TorchModuleWrapper
    torch_stateless_module::PyObject
    dtype::PyObject
    device::PyObject
    params::Tuple
    buffers::Tuple
end

Base.show(io::IO, f::TorchModuleWrapper) = print(io, f.torch_stateless_module, " ", f.dtype, " ", f.device, " params size=", size.(f.params))

Base.length(f::TorchModuleWrapper) = length(f.params)
Base.iterate(f::TorchModuleWrapper) = iterate(f.params)
Base.iterate(f::TorchModuleWrapper, state) = iterate(f.params, state)

function TorchModuleWrapper(torch_module, device)
    pybuiltin("isinstance")(torch_module, torch.nn.Module) || error("Not a torch.nn.Module")
    torch_module = torch_module.to(device)
    funmod, params, buffers = functorch.make_functional_with_buffers(torch_module)
    dtype = params[1].dtype
    #jlparams = map(x -> x.detach().numpy(), params)
    jlparams = map(params) do x
        DLArray(x, pyto_dlpack)
    end
    return TorchModuleWrapper(funmod, dtype, device, jlparams, buffers)
end

function TorchModuleWrapper(torch_module)
    #device = torch.cuda.is_available() ? torch.device("cuda:0") : torch.device("cpu")
    device = torch.device("cpu")
    TorchModuleWrapper(torch_module, device)
end

function (wrap::TorchModuleWrapper)(args...)
    # TODO: handle multiple outputs
    params = wrap.params
    tensor_out = wrap.torch_stateless_module(Tuple(map(x -> torch.as_tensor(x).to(device = wrap.device, dtype = wrap.dtype).requires_grad_(true), params)),
        wrap.buffers, map(x -> torch.as_tensor(PyReverseDims(x)).to(dtype = wrap.dtype, device = wrap.device), args)...)
    res = ReverseDimsArray(DLArray(tensor_out, pyto_dlpack))
    return res
end

function ChainRulesCore.rrule(wrap::TorchModuleWrapper, args...)
    params = wrap.params
    torch_primal, torch_vjpfun = functorch.vjp(py"buffer_implicit"(wrap.torch_stateless_module, wrap.buffers), Tuple(map(x -> torch.as_tensor((x)).to(device = wrap.device, dtype = wrap.dtype).requires_grad_(true), params)),
        map(x -> torch.as_tensor(PyReverseDims(x)).to(dtype = wrap.dtype, device = wrap.device).requires_grad_(true), args)...)
    project = ProjectTo(args)
    function TorchModuleWrapper_pullback(Δ)
        torch_tangent_vals = torch_vjpfun(torch.as_tensor(PyReverseDims(Δ)).to(dtype = wrap.dtype, device = wrap.device))
        jlparams_tangents = map(x -> DLArray(x, pyto_dlpack), torch_tangent_vals[1])
            args_tangents = project(map(x -> ReverseDimsArray(DLArray(x, pyto_dlpack)), torch_tangent_vals[2:end]))
        return (Tangent{TorchModuleWrapper}(; torch_stateless_module = NoTangent(), dtype = NoTangent(), device = NoTangent(), params = jlparams_tangents, buffers = NoTangent()), args_tangents...)
    end
    res = ReverseDimsArray(DLArray(torch_primal, pyto_dlpack))
    return res, TorchModuleWrapper_pullback
end


function __init__()
    try
        copy!(torch, pyimport("torch"))
        copy!(dlpack, pyimport("torch.utils.dlpack"))
        copy!(functorch, pyimport("functorch"))
        copy!(inspect, pyimport("inspect"))
        ispysetup[] = true
        py"""
        def buffer_implicit(fn, buffers):
            def newfn(params, inputs):
                return fn(params, buffers, inputs)
            
            return newfn
        """        
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