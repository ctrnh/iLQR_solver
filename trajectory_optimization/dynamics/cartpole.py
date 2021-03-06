import numpy as np
from .dynamics import Dynamics
class CartPoleDynamics(Dynamics):
    def __init__(self, dt, m, M, l):
        """
        State x = (x, theta, x_dot, theta_dot)
        Theta = 0 (rad) corresponds to pole pointing downwards
        Args:
         - dt: discretization step for the dynamics
         - m: mass of pole
         - M: mass of cart
         - l: length of pole
        """
        self.dt = dt
        self.m = m
        self.M = M
        self.l = l
        self.g = 9.81

        super().__init__(state_dim=4, u_dim=1)



    def f(self, x, u):
        """
        if we denote g by x_dot(t) = g(x(t),u(t))
        Then x_{k+1} = x_k + g(x_k, u_k)dt = f(x_k,u_k)
        """
        u = u[0][0]
        cth = np.cos(x[1][0])
        sth = np.sin(x[1][0])
        theta_dot = x[3][0]

        dxdt = np.zeros((self.state_dim, 1))
        dxdt[0][0] = x[2][0] #x_dot
        dxdt[1][0] = theta_dot

        tmp_div = self.M + self.m*sth**2

        dxdt[2][0] =  (u + self.m*sth*(self.g*cth + self.l*theta_dot**2)) / tmp_div
        dxdt[3][0] = (-(self.m+self.M) * self.g * sth \
                      - self.m*self.l*theta_dot**2*cth*sth \
                      - u*cth )/ (self.l*tmp_div)
        return x + dxdt*self.dt

    def analytical_df_du(self,x,u):
        """
        (useful for tests)
        Implementation of Jacobian of f with respect to u (analytical form)
        """
        cth = np.cos(x[1][0])
        sth = np.sin(x[1][0])
        tmp_div = self.M + self.m*sth**2
        Jac_u = np.zeros((self.state_dim, self.u_dim))
        Jac_u[2][0] = 1/tmp_div
        Jac_u[3][0] = -cth/(self.l*tmp_div)
        return Jac_u*self.dt

    def analytical_df_dx(self, x,u):
        """
        (useful for tests)
        Implementation of Jacobian of f with respect to x (analytical form)
        """
        u = float(u)
        cth = np.cos(x[1][0])
        sth = np.sin(x[1][0])
        theta_dot = x[3][0]
        Jac_x = np.eye((self.state_dim))
        tmp_div = self.M + self.m*sth**2

        Jac_x[0][2] += self.dt
        Jac_x[1][3] += self.dt
        Jac_x[2][1] += (self.m*self.dt/tmp_div**2) * (self.g*(self.M - (self.m+2*self.M)*sth**2)\
                                                      + self.l*theta_dot**2*cth*(self.M - self.m*sth**2)\
                                                      - 2*u*cth*sth)
        Jac_x[2][3] += (2*self.m*self.l*theta_dot*sth*self.dt)/tmp_div

        Jac_x[3][1] += (self.dt/(self.l*tmp_div**2)) * ((self.M+self.m)*self.g*cth*(self.m*sth**2 - self.M)\
                                                      + self.m*self.l*theta_dot**2 * ((self.m+2*self.M)*sth**2 - self.M)\
                                                      + u*sth*(self.M + self.m*(2-sth**2)))

        Jac_x[3][3] += - (2*self.m*theta_dot*cth*sth*self.dt)/tmp_div

        return Jac_x
