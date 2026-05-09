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
        self.backward = jax.jit(self._backward, static_argnums=(2,)) # reg
        self.forward = jit(self._forward)
        self.armijo_condition = jit(self._armijo_condition)
        self.total_cost = jit(self._total_cost)
        self.solve = jax.jit(self._solve, static_argnums=(2, 3, 4, 5, 6))  # n_iter, reg, rho, beta, max_iter
        
    def _backward(
        self,
        xs,
        us,
        reg
    ):
        B, T, nu = us.shape
        nx = self.nx
        
        Vx  = vmap(self.lf_x)(xs[:, -1])
        Vxx = vmap(self.lf_xx)(xs[:, -1])
        
        I_nu = jnp.eye(nu)
        
        # vmap both over B and over T. Flatten it only for vmapping and then reshape it back
        xs_flat = xs[:, :-1].reshape(B*T, nx)   
        us_flat = us.reshape(B*T, nu)     
            
        lx_all, lu_all, lxx_all, luu_all, lux_all = vmap(self.run_derivs)(xs_flat, us_flat)
        fx_all, fu_all = vmap(self.dyn_derivs)(xs_flat, us_flat)
        
        lx_all  = lx_all.reshape(B, T, nx)
        lu_all  = lu_all.reshape(B, T, nu)
        lxx_all = lxx_all.reshape(B, T, nx, nx)
        luu_all = luu_all.reshape(B, T, nu, nu)
        lux_all = lux_all.reshape(B, T, nu, nx)
        fx_all  = fx_all.reshape(B, T, nx, nx)
        fu_all  = fu_all.reshape(B, T, nx, nu)
                
        def backward_scan(carry, inputs):
            Vx, Vxx = carry
            lx, lu, lxx, luu, lux, fx, fu = inputs
            
            Qx = lx + jnp.einsum('bki,bk->bi', fx, Vx) 
            Qu = lu + jnp.einsum('bki,bk->bi', fu, Vx) 
            Qxx = lxx + jnp.einsum('bji,bjk,bkl->bil', fx, Vxx, fx)
            Quu = luu + jnp.einsum('bji,bjk,bkl->bil', fu, Vxx, fu) + I_nu * reg
            Qux = lux + jnp.einsum('bji,bjk,bkl->bil', fu, Vxx, fx)
            
            k = -jnp.linalg.solve(Quu, Qu[:, :, None])[:, :, 0]
            K = -jnp.linalg.solve(Quu, Qux)
            
            Vx_new  = Qx  + jnp.einsum('bji,bj->bi', Qux, k)
            Vxx_new = Qxx + jnp.einsum('bji,bjk->bik', Qux, K)
            
            dV1_t = jnp.einsum('bi,bi->b', k, Qu)
            dV2_t = jnp.einsum('bi,bij,bj->b', k, Quu, k)
            
            return (Vx_new, Vxx_new), (k, K, dV1_t, dV2_t)
        
        # T dimension first, B after
        inputs = (
            lx_all.transpose(1, 0, 2),
            lu_all.transpose(1, 0, 2),
            lxx_all.transpose(1, 0, 2, 3),
            luu_all.transpose(1, 0, 2, 3),
            lux_all.transpose(1, 0, 2, 3),
            fx_all.transpose(1, 0, 2, 3),
            fu_all.transpose(1, 0, 2, 3),
        )
        
        carry = (
            Vx,
            Vxx
        )
        
        (_, _), (ks, Ks, dV1s, dV2s) = jax.lax.scan(backward_scan, carry,  inputs, reverse=True)
        
        dV1 = dV1s.sum(axis=0)
        dV2 = dV2s.sum(axis=0)
        
        return ks.transpose(1, 0, 2), Ks.transpose(1, 0, 2, 3), dV1, dV2
    
    def _solve(
        self,
        x0,  
        us_guess,
        n_iter: int,
        # Backward parameters:
        reg: float = 1e-4,
        # Line search parameters:
        ls_rho: float = 0.5,
        ls_beta: float = 1e-4,
        ls_max_iter: int = 10,
    ):        
        def solve_scan(traj, _):
            xs,us = traj
            ks, Ks, dV1, dV2 = self._backward(xs, us, reg)
            new_xs, new_us = self._forward_ls(xs, us, ks, Ks, dV1, dV2, ls_rho, ls_beta, ls_max_iter)
            return (new_xs, new_us), None

        carry = (
            self._rollout(x0, us_guess),
            us_guess
        )   
        
        (xs,us), _ = jax.lax.scan(solve_scan, carry, None, length=n_iter)
        
        return xs,us
    
    def _forward(
        self,
        xs,
        us,
        ks,
        Ks,
        alpha
    ):
        def forward_scan(x, inputs):
            xs, us, ks, Ks = inputs
            
            delta_x = x - xs
            u = us + alpha[:, None] * ks + jnp.einsum('bij,bj->bi', Ks, delta_x)
            x_new = vmap(self.f)(x, u)
            
            return x_new, (x_new, u)

        inputs = (
            xs[:, :-1].transpose(1,0,2),
            us.transpose(1,0,2),
            ks.transpose(1,0,2),
            Ks.transpose(1,0,2,3)
        )

        _, (xs_1toT, us_new) = jax.lax.scan(forward_scan, xs[:, 0], inputs)
    
        new_xs = jnp.concatenate([xs[:, 0:1, :], xs_1toT.transpose(1, 0, 2)], axis=1)
        new_us = us_new.transpose(1, 0, 2)
        
        return new_xs, new_us
    
    # Forward with Armijo line search
    def _forward_ls(
        self,
        xs,
        us,
        ks,
        Ks,
        dV1,
        dV2,
        rho,
        beta,
        max_iter
    ):
        B, _, _ = xs.shape
        J_nom = self._total_cost(xs, us)
        
        # TODO: no early exit for now, I might want to rewrite it with lax.while_loop
        def forward_ls_scan(carry, _):
            new_xs, new_us, alpha, accepted = carry
            try_xs, try_us = self._forward(xs, us, ks, Ks, alpha)
            armijo_ok = self._armijo_condition(try_xs, try_us, J_nom, alpha, dV1, dV2, beta)
            newly_accepted = armijo_ok & ~accepted
            new_xs = jnp.where(newly_accepted[:, None, None], try_xs, new_xs)
            new_us = jnp.where(newly_accepted[:, None, None], try_us, new_us)
            accepted = accepted | newly_accepted
            alpha = jnp.where(accepted, alpha, alpha * rho)
            return (new_xs, new_us, alpha, accepted), None
            
        carry = (
            xs, 
            us, 
            jnp.ones(B), # alphas
            jnp.zeros(B, dtype=bool) # accepted
        )
        
        (new_xs, new_us, _, _), _ = jax.lax.scan(forward_ls_scan, carry, None, length=max_iter)
        
        return new_xs, new_us
            
    def _armijo_condition(
        self,
        xs,
        us,
        J_nom,
        alpha,
        dV1,
        dV2,
        beta
    ):
        J_try = self._total_cost(xs,us)

        return J_try < J_nom + beta * (alpha * dV1 + 0.5 * alpha**2 * dV2)
    
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
    
    def _total_cost(
        self,
        xs,
        us
    ):
        B, T, _ = us.shape
        
        xs_flat = xs[:, :-1].reshape(B*T, -1)
        us_flat = us.reshape(B*T, -1)
        
        running = vmap(self.l)(xs_flat, us_flat).reshape(B, T).sum(axis=1)
        final = vmap(self.lf)(xs[:,-1])
        
        return running + final