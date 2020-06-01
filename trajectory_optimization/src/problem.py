import numpy as np
class Problem:
    def __init__(self,
                 dynamics,
                 cost,
                 dt,
                 x0,
                 ):
        self.N = cost.N
        self.dt = cost.dt
        self.tf = self.N*self.dt

        self.dyn = dynamics
        self.cost = cost
        self.x0 = x0

        assert self.x0.shape[0] == self.dyn.state_dim
        assert self.x0.shape[1] == 1


        if "state_dim" in self.cost.__dict__:
            assert self.dyn.u_dim == self.cost.u_dim
            assert self.dyn.state_dim == self.cost.state_dim
