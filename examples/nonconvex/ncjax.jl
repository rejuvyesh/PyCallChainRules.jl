using PyCallChainRules.Jax: JaxFunctionWrapper, jax
using PyCall
using LinearAlgebra
using Nonconvex
Nonconvex.@load Ipopt

py"""
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import jax
import jax.numpy as np

def rk4(dynamics, dt=0.01):
  def integrator(x, u, t):
    dt2 = dt / 2.0
    k1 = dynamics(x, u, t)
    k2 = dynamics(x + dt2 * k1, u, t)
    k3 = dynamics(x + dt2 * k2, u, t)
    k4 = dynamics(x + dt * k3, u, t)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
  return integrator

@jax.jit
def acrobot(x, u, t, params=(1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0)):
  # classic acrobot
  del t  # Unused

  m1, m2, l1, lc1, lc2, I1, I2 = params
  g = 9.8
  a = u[0]
  theta1 = x[0]
  theta2 = x[1]
  dtheta1 = x[2]
  dtheta2 = x[3]
  d1 = (
      m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 +
      I2)
  d2 = m2 * (lc2**2 + l1 * lc2 * np.cos(theta2)) + I2
  phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
  phi1 = (-m2 * l1 * lc2 * dtheta2**2 * np.sin(theta2) -
          2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2) +
          (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2)
  ddtheta2 = ((a + d2 / d1 * phi1 -
               m2 * l1 * lc2 * dtheta1**2 * np.sin(theta2) - phi2) /
              (m2 * lc2**2 + I2 - d2**2 / d1))
  ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
  return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2])

dynamics = rk4(acrobot, dt=0.1)
"""

jaxdynamics = JaxFunctionWrapper(py"dynamics")

function dynamics_cost(x, u)
    cost = 0.0
    for i in 1:(size(x, 2)-1)
        nextx = jaxdynamics(x[:, i], u[:, i], [0.1])
        cost = cost + sum(abs2, nextx - x[:, i+1])
    end
    return cost
end

# somewhat dumb objective
function objective(x::AbstractMatrix)
    goal = vec([pi 0.0 0.0 0.0])
    state = x[begin:begin+3, :]
    action = x[begin+4:end, :]
    delta = state .- goal
    term_cost = 0.5 * 1000 * dot(delta, delta)
    eff_cost = 0.5 * 0.01 * dot(action, action)
    dyn_cost = dynamics_cost((state), (action))
    return sum(term_cost + eff_cost + dyn_cost)
end

function objective(x::AbstractVector)
    return objective(reshape(x, (5, :)))
end

T = 10
x0 = vec(rand(5, T))
@info "objective" objective(x0)

model = Model(objective)
# add variable to optimize
addvar!(model, -100*ones(size(x0)), ones(size(x0))*100)
alg = IpoptAlg()
options = IpoptOptions(first_order=true)
result = optimize(model, alg, x0, options=options)
newx = reshape(result.minimizer, (5, :))
@info "objective" objective(newx)