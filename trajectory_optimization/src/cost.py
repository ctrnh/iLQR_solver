import numpy as np

class Cost:
    def __init__(self, goal, N, u_goal):
        self.N = N
        self.goal = goal
        self.u_goal = u_goal

    def loss(self, x, u, k):
        raise NotImplementedError

    def l_x(self, x, k=None, u=None):
        raise NotImplementedError

    def l_u(self, u, k=None, x=None):
        raise NotImplementedError

    def l_xx(self, x=None, u=None, k=None):
        raise NotImplementedError

    def l_uu(self, x=None, u=None, k=None):
        raise NotImplementedError

    def l_xu(self, x=None, u=None, k=None):
        raise NotImplementedError

    def l_ux(self, x=None, u=None, k=None):
        raise NotImplementedError

class QuadraticCost(Cost):
    def __init__(self, dt, N, goal, Q, R, Qf=None, H=None, u_goal=None):
        super().__init__(goal=goal, N=N, u_goal=u_goal)
        self.dt = dt
        self._Q = Q
        self.R = R

        self.Qf = Qf
        if self.Qf is None:
            self.Qf = self.Q


        self.state_dim = self.Q.shape[0]
        self.u_dim = self.R.shape[0]

        self.H = H
        if self.H is None:
            self.H = np.zeros((self.state_dim, self.u_dim))
        if self.u_goal is None:
            self.u_goal = np.zeros((self.u_dim,1))


    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, new_Q):
        assert new_Q.shape[0] == new_Q.shape[0] == self.goal.shape[0]
        self._Q = new_Q

    def loss(self, x, u, k):
        assert k <= self.N
        if k<self.N:
            assert u.shape == self.u_goal.shape, "u: "+str(u.shape) +", ugoal: " + str(self.u_goal.shape)
        if k == self.N:
            l = (1/2)*(x-self.goal).T.dot(self.Qf).dot(x-self.goal)
            return float(l) #final cost: do not multiply by dt

        l = (1/2)*(x-self.goal).T.dot(self.Q).dot(x-self.goal)
        l += (1/2)*(u-self.u_goal).T.dot(self.R).dot(u-self.u_goal)
        l += x.T.dot(self.H).dot(u) +  u.T.dot(self.H.T).dot(x)
        return float(l)*self.dt


    def l_x(self, x,  u, k):
        """
        Derivative of loss wrt x (dl/dx)
        col vector size x_dim
        """
        assert k <= self.N
        assert x.shape[0] == self.state_dim
        if k == self.N:
            lx = self.Qf.dot(x-self.goal)
            return lx

        lx = self.Q.dot(x - self.goal)
        lx += self.H.dot(u)

        assert lx.shape[1] == 1
        assert lx.shape[0] == self.state_dim

        return lx*self.dt


    def l_u(self, x, u, k):
        """ column vector size u_dim """
        assert k < self.N
        assert u.shape[0] == self.u_dim
        lu = self.R.dot(u - self.u_goal)
        lu += self.H.T.dot(x)

        assert lu.shape[1] == 1
        assert lu.shape[0] == self.u_dim
        return lu*self.dt

    def l_xx(self, x, u, k):
        if k == self.N:
            return self.Qf
        lxx = self.Q
        return lxx*self.dt

    def l_uu(self, x, u, k):
        assert k < self.N
        return self.R*self.dt

    def l_xu(self, x, u, k):
        assert k < self.N
        return self.H*self.dt

    def l_ux(self, x, u, k):
        assert k < self.N
        return self.H.T*self.dt
