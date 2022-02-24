
function forward_pass!(s, torchmodel, dummyinput, batchsize, device; samples=100, evals=10)
    ml = torchmodel
    if device == "cuda"
        td = torch.device("cuda:0")
    else
        td = torch.device("cpu")
    end
    ml = ml.to(device=device)
    mlfun, params, buffers = functorch.make_functional_with_buffers(deepcopy(ml))
    mlfun2 = x->mlfun(params, buffers, x)
    mlwrap = TorchModuleWrapper(torchmodel)


    inshape = size(dummyinput)
    ## Torch
    b_torch = @benchmarkable($ml(x), setup=(x=torch.randn($batchsize, $(inshape)..., dtype=torch.float32, device=$td)), samples=samples, evals=evals)

    ## Functorch
    b_functorch = @benchmarkable($mlfun2(x), setup=( x=torch.randn($batchsize, $(inshape)..., dtype=torch.float32, device=$td)), samples=samples, evals=evals)

    ## TorchModuleWrapper
    if device == "cuda"
        b_jl = @benchmarkable($mlwrap(x), setup=(x=CUDA.cu(randn(Float32, $(reverse(inshape))..., $batchsize))), samples=samples, evals=evals)
    else
        b_jl = @benchmarkable($mlwrap(x), setup=(x=randn(Float32, $(reverse(inshape))..., $batchsize)), samples=samples, evals=evals)
    end
    # @info "Average slowdown vs torch" (mean(b_jl.times) / mean(b_torch.times))
    # @info "Average slowdown vs functorch" (mean(b_jl.times) / mean(b_functorch.times))

    # return b_torch, b_functorch, b_jl
    s["torch"] = b_torch
    s["functorch"] = b_functorch
    s["jl"] = b_jl
end

function backward_pass!(s, torchmodel, dummyinput, batchsize, device; samples=100, evals=10, seconds=20)
    ml = torchmodel
    if device == "cuda"
        td = torch.device("cuda:0")
    else
        td = torch.device("cpu")
    end
    ml = ml.to(device=device)
    mlfun, params, buffers = functorch.make_functional_with_buffers(deepcopy(ml))
    mlfun2 = x->mlfun(params, buffers, x)
    mlwrap = TorchModuleWrapper(torchmodel)

    inshape = size(dummyinput)

    ## Torch
    torchgrad = (m, y)->torch.autograd.grad(m(y).sum(), m.parameters())
    ## Functorch
    funcgrad = (params, y) -> functorch.grad(mlfun(params, buffers, y).sum(), params)
    ## TorchModuleWrapper
    zygotegrad = (m, y) -> Zygote.gradient(m->sum(m(y)), m)

    b_torch = @benchmarkable($torchgrad(m, x), setup=(m=$(torchmodel); x=torch.randn($batchsize, $(inshape)..., dtype=torch.float32, device=$td)), samples=samples, evals=evals, seconds=seconds)

    b_functorch = @benchmarkable($funcgrad(params, x), setup=(params=$params; x=torch.randn($batchsize, $(inshape)..., dtype=torch.float32, device=$td)), samples=samples, evals=evals, seconds=seconds)

    if device == "cuda"
        b_jl = @benchmarkable($zygotegrad(mlwrap, x), setup=(mlwrap=$mlwrap; x=CUDA.cu(randn(Float32, $(reverse(inshape))..., $batchsize))), samples=samples, evals=evals, seconds=seconds)
    else
        b_jl = @benchmarkable($zygotegrad(mlwrap, x), setup=(mlwrap=$mlwrap; x=randn(Float32, $(reverse(inshape))..., $batchsize)), samples=samples, evals=evals, seconds=seconds)
    end
    # @info "Average slowdown vs torch" (mean(b_jl.times) / mean(b_torch.times))
    # @info "Average slowdown vs functorch" (mean(b_jl.times) / mean(b_functorch.times))

    # return b_torch, b_functorch, b_jl
    s["torch"] = b_torch
    s["functorch"] = b_functorch
    s["jl"] = b_jl
end