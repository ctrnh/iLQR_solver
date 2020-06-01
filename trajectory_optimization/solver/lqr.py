import numpy as np 
class LQR(Solver):
    def __init__(self, LinearProblem):
        super().__init__(LinearProblem)

    def init(self):
        self.J_star = [None for i in range(self.LP.N + 1)]

        self.V = [None for i in range(self.LP.N + 1)]
        self.W = [None for i in range(self.LP.N + 1)]


        self.H = [self.LP.cost.hess_l1(self.goal(i), i) for i in range(self.LP.N + 1)]
        self.G = [self.LP.cost.grad_l1(self.goal(i), i) for i in range(self.LP.N + 1)]

        self.L = [None for i in range(self.LP.N)]


    def goal(self, n):
        if self.LP.cost.unique_goal:
            return self.LP.cost.goal
        return self.LP.cost.goal[n]

    def backward(self):
        N = self.LP.N
        R = self.LP.cost.R
        A = self.LP.dyn.A
        B = self.LP.dyn.B

        self.V[N] = (1/2) * self.H[N]
        self.W[N] = self.G[N] - self.H[N].dot(self.goal(N))


        for i in range(self.LP.N, 0, -1):
            tmp = A.T.dot(self.V[i]).dot(B)
            tmp2 = np.linalg.inv(R+B.T.dot(self.V[i]).dot(B))
            self.V[i-1] = .5 * self.H[i-1] - tmp.dot(tmp2).dot(tmp.T) + A.T.dot(self.V[i]).dot(A)

            self.W[i-1] = self.G[i-1] - self.H[i-1].T.dot(self.goal(i-1)) + A.T.dot(self.W[i]) \
            - tmp.dot(tmp2).dot(B.T.dot(self.W[i]))

    def forward(self):
        """
        Compute lqr controls u, states x, J_star
        """
        R = self.LP.cost.R
        A = self.LP.dyn.A
        B = self.LP.dyn.B

        x = self.LP.x0
        self.x[0] = x
        for i in range(self.LP.N):
            u = - np.linalg.inv(R+B.T.dot(self.V[i+1]).dot(B)).dot(.5*B.T.dot(self.W[i+1]) \
                                                           + B.T.dot(self.V[i+1]).dot(A).dot(x))
            if self.LP.dyn.u_dim == 1:
                self.u[i] = float(u)
            else:
                self.u[i] = u
            self.J_star[i] = float(x.T.dot(self.V[i]).dot(x) + self.W[i].T.dot(x)) #up to constant

            if i == 0:
                self.J[i] =  self.LP.cost.loss(x, u, i)
            else:
                self.J[i] = self.J[i-1] + self.LP.cost.loss(x, u, i)
            x = self.LP.dyn.next_state(x, u)
            self.x[i+1] = x

        self.J[self.LP.N] = self.J[self.LP.N-1] + self.LP.cost.loss(x, 0, self.LP.N)

        self.J_star[self.LP.N] = float(x.T.dot(self.V[self.LP.N]).dot(x) \
                                       + self.W[self.LP.N].T.dot(x)) #up to constant

    def solve(self):
        self.init()
        self.backward()
        self.forward()
        self.solved = True
        print("Last Loss: ", self.J_star[-1])
