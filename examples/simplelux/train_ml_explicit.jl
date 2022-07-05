using Lux
using Optimisers
using Random
using PyCallChainRules.Jax: LuxStaxWrapper, jax, stax
using Zygote

input_dim = 4
output_dim = 2
hiddendim = 16
batchsize = 6

jax_init_fun, jax_apply_fun = stax.serial(stax.Dense(hiddendim), stax.Relu, 
                                        stax.Dense(hiddendim), stax.Relu, 
                                        stax.Dense(output_dim))

jlmodel = LuxStaxWrapper(jax_init_fun, jax.jit(jax_apply_fun); input_shape=(-1, input_dim))

rng = Random.default_rng()

ps, st = Lux.setup(rng, jlmodel)


input = randn(Float32, input_dim, batchsize) |> Lux.gpu
target = randn(Float32, output_dim, batchsize) |> Lux.gpu

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