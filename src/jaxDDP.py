import jax
from jax import grad, jacrev, vmap, jit
import jax.numpy as jnp

class JaxDDP:
    def __init__(self, dynamics, running_cost, terminal_cost, nx: int, nu: int):
        self.f  = dynamics
        self.l  = running_cost
        self.lf = terminal_cost
        self.nx = nx
        self.nu = nu

        self.f_x = jacrev(self.f, argnums=0)
        self.f_u = jacrev(self.f, argnums=1)

        self.l_x  = grad(running_cost, argnums=0)
        self.l_u  = grad(running_cost, argnums=1)

        self.l_xx = jacrev(self.l_x, argnums=0)
        self.l_uu = jacrev(self.l_u, argnums=1)
        self.l_ux = jacrev(self.l_u, argnums=0)

        self.lf_x  = grad(terminal_cost)
        self.lf_xx = jacrev(grad(terminal_cost))
        
        self.run_derivs = lambda x, u: (self.l_x(x,u), self.l_u(x,u), self.l_xx(x,u), self.l_uu(x,u), self.l_ux(x,u))
        self.dyn_derivs = lambda x, u: (self.f_x(x,u), self.f_u(x,u))

        self.rollout  = jit(self._rollout)
    
    def _rollout(
        self,
        x0, 
        us
    ):
        def rollout_scan(x, u):
            x_next = vmap(self.f)(x, u)
            return x_next, x_next

        _, xs_1toT = jax.lax.scan(rollout_scan, x0, us.transpose(1, 0, 2)) # (T, B, nx)

        xs = jnp.concatenate([x0[:, None, :], xs_1toT.transpose(1, 0, 2)], axis=1) # (B, T+1, nx)
        return xs