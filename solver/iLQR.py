class iLQR(Solver):
    def __init__(self,
                 problem,
                 n_traj,
                 u_init,
                 alpha_init=1,
                 crit_alpha=1e-1,
                rho_reg=0,
                beta=[1e-4,10],
                gamma=.5,
                alpha_min=1e-3):
        """
        n_traj (int) : number of trajectories
        u_init (list[float]) : sequence of controls (N elements)

        """

        super().__init__(problem)
        self.n_traj = n_traj
        self.u_init = u_init


        self.rho_reg = rho_reg

        # Line search param
        self.alpha_init = alpha_init
        self.alpha = alpha_init
        self.crit_alpha = crit_alpha
        self.alpha_min = alpha_min

        self.gamma = gamma
        self.beta = beta
        assert len(self.u_init) == self.LP.N
        assert self.u_init[0].shape[1] == 1
        assert self.u_init[0].shape[0] == self.LP.dyn.u_dim

    def reset(self):
        #self.V = [None for i in range(self.LP.N + 1)] # "S_k"
        #self.w = [None for i in range(self.LP.N + 1)] # "s_k"

        self.K = [None for i in range(self.LP.N)] # (K_0, ..., K_N-1) for each u_k
        self.d = [None for i in range(self.LP.N)]
        self.cond_nb = []
        self.expected_cost = [None for i in range(self.LP.N)]
        self.J = [None for i in range(self.LP.N+1)]
        #self.du = [None for i in range(self.LP.N)]
        #self.dx =  [None for i in range(self.LP.N +1)]
        #self.dx[0] = np.zeros_like(self.LP.x0)

    def backward(self):
        #self.reset()

        #self.w[self.LP.N] = self.LP.cost.l_x(x=self.x[self.LP.N],u=None, k=self.LP.N)
        #self.V[self.LP.N] = self.LP.cost.l_xx(x=self.x[self.LP.N],u=None, k=self.LP.N)
        w = self.LP.cost.l_x(x=self.x[self.LP.N],u=None, k=self.LP.N)
        V = self.LP.cost.l_xx(x=self.x[self.LP.N],u=None, k=self.LP.N)
        assert np.abs(V - self.LP.cost.Qf).all() <= 1e-3
        assert np.abs(self.LP.cost.Qf.dot(self.x[self.LP.N] \
                                          - self.LP.cost.goal) \
                      - w).all() <= 1e-3, str(w) + " " + str(self.LP.cost.Qf.dot(self.x[self.LP.N] \
                                          - self.LP.cost.goal))


        for k in range(self.LP.N-1, -1, -1):
            A = self.LP.dyn.df_dx(x=self.x[k],u=self.u[k])
            B = self.LP.dyn.df_du(x=self.x[k],u=self.u[k])

            Qx = self.LP.cost.l_x(x=self.x[k], u=self.u[k],k=k) + A.T.dot(w)#A.T.dot(self.w[k+1])
            assert Qx.shape[0] == self.LP.dyn.state_dim
            Qu = self.LP.cost.l_u(x=self.x[k], u=self.u[k],k=k) + B.T.dot(w)#B.T.dot(self.w[k+1])
            assert Qu.shape[0] == self.LP.dyn.u_dim
            assert Qx.shape[1] == Qu.shape[1] == 1

            Qux = self.LP.cost.l_ux(x=self.x[k], u=self.u[k],k=k) + B.T.dot(V).dot(A)#B.T.dot(self.V[k+1]).dot(A)
            assert Qux.shape[0] == self.LP.dyn.u_dim
            assert Qux.shape[1] == self.LP.dyn.state_dim

            Qxu = Qux.T
            Qxx = self.LP.cost.l_xx(x=self.x[k], u=self.u[k],k=k) + A.T.dot(V).dot(A)#A.T.dot(self.V[k+1]).dot(A)
            Quu = self.LP.cost.l_uu(x=self.x[k], u=self.u[k],k=k) + B.T.dot(V).dot(B)#B.T.dot(self.V[k+1]).dot(B)
            Quu += self.rho_reg*np.eye(Quu.shape[0])
            assert Quu.shape[0] == Quu.shape[1] == self.LP.dyn.u_dim
            assert Qxx.shape[0] == Qxx.shape[1] == self.LP.dyn.state_dim

            Quu_inv = np.linalg.inv(Quu)


            cond_nb_Quu = np.linalg.cond(Quu)


            self.K[k] = - Quu_inv.dot(Qux)
            assert self.K[k].shape[0] == self.LP.dyn.u_dim
            assert self.K[k].shape[1] == self.LP.dyn.state_dim
            self.d[k] = - Quu_inv.dot(Qu)
            assert self.d[k].shape[1] == 1
            assert self.d[k].shape[0] == self.LP.dyn.u_dim

            self.expected_cost[k] = [self.d[k].T.dot(Qu),
                                     .5* self.d[k].T.dot(Quu).dot(self.d[k])]

            w = Qx + self.K[k].T.dot(Qu) + (self.K[k].T.dot(Quu) + Qxu).dot(self.d[k])
            V = Qxx + self.K[k].T.dot(Quu).dot(self.K[k]) + Qxu.dot(self.K[k]) + self.K[k].T.dot(Qux)
            #self.w[k] = Qx + self.K[k].T.dot(Qu) + (self.K[k].T.dot(Quu) + Qxu).dot(self.d[k])
            #self.V[k] = Qxx + self.K[k].T.dot(Quu).dot(self.K[k]) + Qxu.dot(self.K[k]) + self.K[k].T.dot(Qux)
            #assert self.w[k].shape[0] == self.V[k].shape[0] == self.V[k].shape[1] == self.LP.dyn.state_dim
            #assert self.w[k].shape[1] == 1

    def forward(self, init=False):

        if init: # First rollout
            self.u = self.u_init.copy()
            self.x[0] = self.LP.x0
            self.J[0] = self.LP.cost.loss(self.x[0], self.u[0],k=0)
            for k in range(self.LP.N):
                self.x[k+1] = self.LP.dyn.f(self.x[k], self.u[k])
                if k == self.LP.N-1:
                    self.J[self.LP.N] = self.J[k] + self.LP.cost.loss(self.x[k+1],
                                                            u=None,
                                                            k=k+1)
                else:
                    self.J[k+1] = self.J[k] + self.LP.cost.loss(self.x[k+1],
                                                            self.u[k+1],
                                                            k=k+1)

            self.init_rollout = self.x

        # si traj_{k+1}- trak{k} > 1: alpha/=2
        #zk+1-zk > c 10-2 alpha/=2, alpha = 10, alpha =1

        else:
            counter = 0
            alpha_ok = False
            Delta_V = 0
            while not alpha_ok :
                counter += 1
                if counter == 2000: assert 1==0

                candidate_u = self.u.copy()
                candidate_x = self.x.copy()
                candidate_J = [None for i in range(self.LP.N+1)]

                dx =  np.zeros((self.LP.dyn.state_dim, 1))
                alpha_ok = True
                for k in range(self.LP.N):
                    du = self.alpha*self.d[k] + self.K[k].dot(dx) #delta u_k

                    dx_2 = (self.LP.dyn.df_dx(candidate_x[k], candidate_u[k]).dot(dx) + \
                            self.LP.dyn.df_du(candidate_x[k], candidate_u[k]).dot(du)) # Pour tester, = delta x_{k+1}

                    candidate_u[k] = candidate_u[k] + du #u_k^t = u_k^{t-1} + delta u_k
                    candidate_x[k] = candidate_x[k] + dx #x_k^t = x_k^{t-1} + delta x_k
                    next_x = self.LP.dyn.f(candidate_x[k], candidate_u[k]) #x_{k+1}^t

                    dx = next_x - candidate_x[k+1] # delta x_{k+1} = x_{k+1}^t - x_{k+1}^{t-1}

                    diff_dx = np.abs(dx-dx_2).flatten()



                    if k == 0:
                        candidate_J[0] = self.LP.cost.loss(candidate_x[k], candidate_u[k],k=k)
                    else:
                        candidate_J[k] = candidate_J[k-1] + self.LP.cost.loss(candidate_x[k], candidate_u[k],k=k)


                    if np.max(diff_dx) >= self.crit_alpha:
                        print(f"k = {k}, alpha = {self.alpha}")
                        print(f"dx = {dx.flatten()}, \n should be equal to \n {dx_2.flatten()} \n")



                    #print(f"dx={dx}, du={du}")
                   # if (np.linalg.norm(dx)/(np.linalg.norm(next_x)+1) >= self.crit_alpha \
                    #    and np.linalg.norm(du)/(np.linalg.norm(tmp_u[k])+1) >= self.crit_alpha \
                     #  and self.alpha >= 1e-3):

                    #if (np.abs(dx) > self.crit_alpha).any() or (np.abs(du) > self.crit_alpha).any():
                    #Delta_V += (self.alpha*self.expected_cost[k][0] + \
                     #           self.alpha**2 * self.expected_cost[k][1])

                    #if ( self.alpha > self.alpha_min and \
                     #   (self.beta[0] >= np.abs((candidate_J[k] - self.J[k])/ Delta_V) or \
                      #  np.abs((candidate_J[k] - self.J[k])/ Delta_V) >= self.beta[1])) :
                        self.alpha = self.gamma*self.alpha
                        alpha_ok = False
                        break


                if alpha_ok:

                    candidate_x[self.LP.N] = candidate_x[self.LP.N] + dx
                    candidate_J[self.LP.N] = candidate_J[self.LP.N-1] + self.LP.cost.loss(candidate_x[self.LP.N],
                                                                                          u=None,
                                                                                          k=self.LP.N)

                    self.x = candidate_x.copy()
                    self.u = candidate_u.copy()
                    self.J = candidate_J
                    #self.alpha = self.alpha_init



    def compute_loss(self):
        self.J[0] = self.LP.cost.loss(self.x[0], self.u[0],k=0)

        for k in range(1,self.LP.N):
            self.J[k] = self.J[k-1] + self.LP.cost.loss(self.x[k], self.u[k],k=k)
        self.J[self.LP.N] = self.J[self.LP.N-1] + self.LP.cost.loss(self.x[self.LP.N], 0,k=self.LP.N)

    def solve(self, ):
        self.reset()
        self.rollout_hist = []
        self.all_cond_nb = []

        self.forward(init=True)

        for i_traj in range(self.n_traj):
            if i_traj%3 == 0:
                self.rollout_hist.append(self.x)
            if i_traj%10==0:
                print(f'**** {i_traj}-th Trajectory ' )

            self.backward()
            self.forward()

            self.all_cond_nb.append(self.cond_nb)


        self.compute_loss()
        self.solved = True
        print("Last Loss: ", self.J[-1])
