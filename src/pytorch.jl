module Torch

using PyCall

using ChainRulesCore
using DLPack
using Functors: @functor
using Adapt


using ..PyCallChainRules: PyAdaptor

const inspect = PyNULL()
const torch = PyNULL()
const functorch = PyNULL()
const dlpack = PyNULL()
const ispysetup = Ref{Bool}(false)

pyto_dlpack(x) = @pycall dlpack.to_dlpack(x)::PyObject
pyfrom_dlpack(x) = @pycall dlpack.from_dlpack(x)::PyObject


struct TorchModuleWrapper
    torch_stateless_module::PyObject
    dtype::PyObject
    params::Tuple
    buffers::Tuple
end

@functor TorchModuleWrapper (params,)

Base.show(io::IO, f::TorchModuleWrapper) = print(io, f.torch_stateless_module, " ", f.dtype, " ", " params size=", size.(f.params))

Base.length(f::TorchModuleWrapper) = length(f.params)
Base.iterate(f::TorchModuleWrapper) = iterate(f.params)
Base.iterate(f::TorchModuleWrapper, state) = iterate(f.params, state)

function TorchModuleWrapper(torch_module)
    pybuiltin("isinstance")(torch_module, torch.nn.Module) || error("Not a torch.nn.Module")
    funmod, params, buffers = functorch.make_functional_with_buffers(torch_module)
    dtype = params[1].dtype
    jlparams = map(params) do x
        DLPack.wrap(x, pyto_dlpack)
    end
    return TorchModuleWrapper(funmod, dtype, jlparams, buffers)
end


function (wrap::TorchModuleWrapper)(args...)
    # TODO: handle multiple outputs
    params = wrap.params
    tensor_out = wrap.torch_stateless_module(Tuple(map(x -> DLPack.share(x, PyObject, pyfrom_dlpack).requires_grad_(true), params)),
        wrap.buffers, map(x -> DLPack.share(x, PyObject, pyfrom_dlpack), args)...)
    res = DLPack.wrap(tensor_out, pyto_dlpack)
    return res
end

function ChainRulesCore.rrule(wrap::TorchModuleWrapper, args...)
    T = typeof(first(args))
    params = wrap.params
    torch_primal, torch_vjpfun = functorch.vjp(py"buffer_implicit"(wrap.torch_stateless_module, wrap.buffers), Tuple(map(x -> DLPack.share(x, PyObject, pyfrom_dlpack).requires_grad_(true), params)),
        map(x -> DLPack.share(x, PyObject, pyfrom_dlpack).requires_grad_(true), args)...)
    project = ProjectTo(args)
    function TorchModuleWrapper_pullback(Δ)
        torch_tangent_vals = torch_vjpfun(DLPack.share(Adapt.adapt_storage(PyAdaptor{T}, Δ), PyObject, pyfrom_dlpack))
        jlparams_tangents = map(x -> (DLPack.wrap(x, pyto_dlpack)), torch_tangent_vals[1])
        args_tangents = project(map(x -> (DLPack.wrap(x, pyto_dlpack)), torch_tangent_vals[2:end]))
        return (Tangent{TorchModuleWrapper}(; torch_stateless_module = NoTangent(), dtype = NoTangent(), params = jlparams_tangents, buffers = NoTangent()), args_tangents...)
    end
    res = DLPack.wrap(torch_primal, pyto_dlpack)
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