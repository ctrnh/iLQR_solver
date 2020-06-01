import numpy as np

class LinearDynamics(Dynamics):
    def __init__(self, A, B):
        self.A = A
        self.B = B

        self.state_dim = self.A.shape[1]
        self.u_dim = self.B.shape[1]
        assert self.B.shape[0] == self.state_dim

    def f(self,x,u):
        return self.A.dot(x) + self.B.dot(u)

    def df_dx(self, x, u):
        return self.A

    def df_du(self, x, u):
        return self.B
