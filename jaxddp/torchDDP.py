import torch
from torch.func import grad, jacrev

class TorchDDP:
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

    def solve(
        self,
        x0: torch.Tensor,  
        us_guess: torch.Tensor,
        n_iter: int,
        # Backward parameters:
        reg: float = 1e-4,
        # Line search parameters:
        ls_rho: float = 0.5,
        ls_beta: float = 1e-4,
        ls_max_iter: int = 10,
    ):
        us = us_guess.clone()
        xs = self.rollout(x0, us)

        for _ in range(n_iter):
            ks, Ks, dV1, dV2 = self.backward(xs, us, reg)
            xs, us = self.forward_ls(xs, us, ks, Ks, dV1, dV2, ls_rho, ls_beta, ls_max_iter)


        return xs, us

    def backward(
        self,
        xs,
        us,
        reg
    ):
        B, T, nu = us.shape
        nx = self.nx
        device = xs.device
        dtype = xs.dtype
 
        I_nu = torch.eye(nu, device=device, dtype=dtype)

        Ks = torch.zeros(B, T, nu, nx, device=device, dtype=dtype)
        ks = torch.zeros(B, T, nu, device=device, dtype=dtype)

        Vx  = torch.vmap(self.lf_x)(xs[:, -1])
        Vxx = torch.vmap(self.lf_xx)(xs[:, -1])

        # vmap both over B and over T. Flatten it only for vmapping and then reshape it back
        xs_flat = xs[:, :-1].reshape(B*T, nx)   
        us_flat = us.reshape(B*T, nu)        

        lx_all, lu_all, lxx_all, luu_all, lux_all = torch.vmap(self.run_derivs)(xs_flat, us_flat)
        fx_all, fu_all = torch.vmap(self.dyn_derivs)(xs_flat, us_flat)

        lx_all  = lx_all.reshape(B, T, nx)
        lu_all  = lu_all.reshape(B, T, nu)
        lxx_all = lxx_all.reshape(B, T, nx, nx)
        luu_all = luu_all.reshape(B, T, nu, nu)
        lux_all = lux_all.reshape(B, T, nu, nx)
        fx_all  = fx_all.reshape(B, T, nx, nx)
        fu_all  = fu_all.reshape(B, T, nx, nu)

        # Armijo expected dV components
        dV1 = torch.zeros(B, device=device, dtype=dtype)
        dV2 = torch.zeros(B, device=device, dtype=dtype)

        for i in reversed(range(T)):           
            lx, lu, lxx, luu, lux = lx_all[:,i], lu_all[:,i], lxx_all[:,i], luu_all[:,i], lux_all[:,i]
            fx, fu = fx_all[:,i], fu_all[:,i]

            Qx = lx + torch.einsum('bki,bk->bi', fx, Vx) 
            Qu = lu + torch.einsum('bki,bk->bi', fu, Vx) 
            Qxx = lxx + torch.einsum('bji,bjk,bkl->bil', fx, Vxx, fx)
            Quu = luu + torch.einsum('bji,bjk,bkl->bil', fu, Vxx, fu) + I_nu * reg
            Qux = lux + torch.einsum('bji,bjk,bkl->bil', fu, Vxx, fx)

            k = -torch.linalg.solve(Quu, Qu)
            K = -torch.linalg.solve(Quu, Qux)

            Vx  = Qx  + torch.einsum('bji,bj->bi', Qux, k)
            Vxx = Qxx + torch.einsum('bji,bjk->bik', Qux, K)

            ks[:, i] = k
            Ks[:, i] = K

            dV1 += torch.einsum('bi,bi->b', k, Qu)
            dV2 += torch.einsum('bi,bij,bj->b', k, Quu, k)

        return ks, Ks, dV1, dV2

    def forward(
        self,
        xs,
        us,
        ks,
        Ks,
        alpha
    ):
        B, T, nu = us.shape
        nx = self.nx
        device = xs.device
        dtype = xs.dtype

        x = xs[:, 0].clone()
        
        new_xs = torch.zeros(B, T+1, nx, device=device, dtype=dtype)
        new_us = torch.zeros(B, T, nu, device=device, dtype=dtype)
        new_xs[:, 0] = x

        for i in range(T):
            delta_x = x - xs[:, i]
            u = us[:,i] + alpha.unsqueeze(-1) * ks[:,i] + torch.einsum('bij,bj->bi', Ks[:, i], delta_x)
            x = torch.vmap(self.f)(x, u)
            new_xs[:, i+1] = x
            new_us[:, i] = u

        return new_xs, new_us
    
    # Forward with Armijo line search
    def forward_ls(
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
        assert rho < 1 and rho > 0
        B, _, _ = xs.shape
        device = xs.device
        dtype = xs.dtype

        alpha = torch.ones(B, device=device, dtype=dtype)
        J_nom = self.total_cost(xs, us)

        new_xs, new_us = xs, us
        accepted = torch.zeros(B, device=device, dtype=torch.bool)

        for _ in range(max_iter):
            try_xs, try_us = self.forward(xs, us, ks, Ks, alpha)

            all_accepted = self.armijo_condition(try_xs, try_us, J_nom, alpha, dV1, dV2, beta)
            newly_accepted = all_accepted & ~accepted

            new_xs = torch.where(newly_accepted.view(B,1,1), try_xs, new_xs)
            new_us = torch.where(newly_accepted.view(B,1,1), try_us, new_us)

            accepted = accepted | newly_accepted

            if accepted.all():
                break

            alpha = torch.where(accepted, alpha, alpha * rho)

        return new_xs, new_us

    def armijo_condition(
        self,
        xs,
        us,
        J_nom,
        alpha,
        dV1,
        dV2,
        beta
    ):
        J_try = self.total_cost(xs,us)

        return J_try < J_nom + beta * (alpha * dV1 + 0.5 * alpha**2 * dV2)

    def rollout(
            self,
            x0, 
            us
    ):
        B, T, _ = us.shape
        nx = self.nx
        device = us.device
        dtype = us.dtype

        xs = torch.zeros(B, T+1, nx, device=device, dtype=dtype)
        x = x0.clone()
        xs[:, 0] = x

        for i in range(T):
            u = us[:, i]
            x = torch.vmap(self.f)(x, u)
            xs[:, i+1] = x

        return xs
    
    def total_cost(
        self,
        xs,
        us
    ):
        B, T, _ = us.shape
        device = us.device
        dtype = us.dtype
        cost = torch.zeros(B, device=device, dtype=dtype)
        
        for i in range(T):
            x = xs[:,i]
            u = us[:,i]

            cost += torch.vmap(self.l)(x,u)

        x = xs[:,-1]
        cost += torch.vmap(self.lf)(x)

        return cost