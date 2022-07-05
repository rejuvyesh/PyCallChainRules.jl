using Lux
using Optimisers
using Random
using Zygote
using PyCallChainRules.Jax: LuxStaxWrapper, jax, stax

# Note when mixing jax and julia layers, recommended to set
# XLA_PYTHON_CLIENT_PREALLOCATE=false

input_dim = 4
output_dim = 2
hiddendim = 16
batchsize = 6

rng = Random.default_rng()

input = randn(rng, Float32, input_dim, batchsize) |> Lux.gpu
target = randn(rng, Float32, output_dim, batchsize) |> Lux.gpu


jax_init_fun, jax_apply_fun = stax.serial(stax.Dense(hiddendim), stax.Relu, 
                                        stax.Dense(hiddendim), stax.Relu, 
                                        stax.Dense(output_dim), stax.Relu)


# Mix of Lux layers and Jax stax layers
# Note: Lux's optimization don't play well
jlmodel = Chain(Dense(input_dim, input_dim, Lux.relu), 
                LuxStaxWrapper(jax_init_fun, jax.jit(jax_apply_fun); input_shape=(batchsize, input_dim)), 
                Dense(output_dim, output_dim); disable_optimizations=true)

ps, st = Lux.setup(rng, jlmodel) .|> Lux.gpu



loss(model, x, y, ps, st) = sum(abs2, Lux.apply(model, x, ps, st)[1] .- y)

@info "before" loss(jlmodel, input, target, ps, st)

function train(model, ps; nsteps=100)
    opt = Optimisers.ADAM(0.01)
    state = Optimisers.setup(opt, ps)
    for i in 1:nsteps
        gs, _ = gradient(ps, input, target) do p, x, y
            loss(model, x, y, p, st)
        end
        state, ps = Optimisers.update(state, ps, gs)
    end
    return ps
end

newps = train(jlmodel, ps)

@info "after" loss(jlmodel, input, target, newps, st)