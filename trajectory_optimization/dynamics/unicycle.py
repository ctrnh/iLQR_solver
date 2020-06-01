import numpy as np
from .dynamics import Dynamics

class UnicycleDynamics(Dynamics):
    def __init__(self, dt):
        """
        state (x,y,v,theta)
        control (v, theta)
        theta = 0 : towards x > 0
        """
        self.dt = dt
        super().__init__(state_dim=4, u_dim=2)



    def f(self, x, u):
        v = x[2][0]
        theta = x[3][0]

        dxdt = np.array([[v*np.cos(theta)], [v*np.sin(theta)], [0], [0]]) \
               + np.array([[0], [0], [u[0][0]], [u[1][0]]])
        #dxdt = np.zeros((self.state_dim, self.u_dim))
        #dxdt[2][0] = u[0][0]
        #dxdt[3][1] = u[1][0]
        return x + dxdt*self.dt
