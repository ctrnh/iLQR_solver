import numpy as np

class Dynamics:
    def __init__(self, state_dim, u_dim):
        self.state_dim = state_dim
        self.u_dim = u_dim

    def f(self, x, u):
        """
        x_{k+1} = self.f(x_k,u_k)
        """
        raise NotImplementedError

    def df_dx(self, x, u):
        """
        Computes the Jacobian of f wrt x using finite difference.
        Args:
         - x : vector of size (state_dim, 1)
         - u : vector of size (u_dim, 1)
        Out:
         - Jac_x array of size (state_dim, state_dim)
        """
        eps=1e-4

        Jac_x = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.state_dim):
            x_h = x.copy().astype(float)
            x_h[i][0] += eps
            Jac_x[:,i] = (self.f(x_h,u) - self.f(x,u)).flatten()

        Jac_x /= eps
        return Jac_x

    def df_du(self, x, u):
        """
        Computes the Jacobian of f wrt u using finite difference.
        Args:
         - x : vector of size (state_dim, 1)
         - u : vector of size (u_dim, 1)
        Out:
         - Jac_u array of size (state_dim, u_dim)
        """
        eps=1e-4
        Jac_u = np.zeros((self.state_dim, self.u_dim))
        for i in range(self.u_dim):
            u_h = u.copy().astype(float)
            u_h[i][0] += eps
            Jac_u[:,i] = (self.f(x,u_h) - self.f(x,u)).flatten()
        Jac_u /= eps
        return Jac_u

    def test_df_dx(self, x, u):
        """
        (TEST : to remove)
        Same as df_dx. But df_dx can be overriden. Jacobian of f wrt x
        """
        eps=1e-4

        Jac_x = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.state_dim):
            x_h = x.copy().astype(float)
            x_h[i][0] =  x_h[i][0] + eps
            Jac_x[:,i] = (self.f(x_h,u) - self.f(x,u)).flatten()
        Jac_x /= eps
        return Jac_x

    def test_df_du(self, x, u):
        """
        Jacobian of f wrt u
        """
        eps=1e-4
        Jac_u = np.zeros((self.state_dim, self.u_dim))
        for i in range(self.u_dim):
            u_h = u.copy().astype(float)
            u_h[i][0] += eps
            Jac_u[:,i] = (self.f(x,u_h) - self.f(x,u)).flatten()
        Jac_u /= eps
        return Jac_u
