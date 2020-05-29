class Environment:
    def __init__(self, dt, state_dim, u_dim):
        self.dt = dt

        self.state_dim = state_dim
        self.u_dim = u_dim

        self.dynamics = None


    def create_problem(self,
                       x0,
                       xf,
                       N,
                       Q=None,
                       R=None,
                       Qf=None,
                      scale=1):
        scale = scale
        if Q is None:
            Q = scale*np.eye(self.state_dim)
        if R is None:
            R = scale*np.eye(self.u_dim)
        cost = QuadraticCost(Q=Q,
                             R=R,
                             Qf=Qf,
                             goal=xf,
                             N=N,
                             dt=self.dt,
                            )
        print("Initial cost: x0, u= 0, J = ", cost.loss(x=x0,u=np.zeros((self.u_dim,1)),k=0))
        print("Last cost: x0, u= 0, J = ", cost.loss(x=x0,u=np.zeros((self.u_dim,1)),k=N))
        return Problem(dynamics=self.dynamics,
                         cost=cost,
                         dt=self.dt,
                         x0=x0
                        )

    def solve_iLQR(self,
                   x0,
                   xf,
                   N,
                   n_traj,
                   alpha_init,
                   crit_alpha,
                   Q=None,
                   R=None,
                   Qf=None,
                   scale_Q=1
                  ):
        """
        solve with iLQR:
        - initialize with random uniform * 1e-3
        - Q, R = eye
        """
        problem = self.create_problem(x0=x0,
                                      xf=xf,
                                      N=N,
                                     Q=Q,
                                     R=R,
                                     Qf=Qf,
                                     scale=scale_Q)
        self.ilqr = iLQR(problem=problem,
                           n_traj=n_traj,
                           u_init=[(1e-3)*np.random.uniform(size=(self.u_dim,1)) for i in range(N)],
                        alpha_init=alpha_init,
                        crit_alpha=crit_alpha)
        self.ilqr.solve()

    def position(self,x):
        raise NotImplementedError


    def create_animation(self, name="animation", x=None):
        if x is None:
            assert self.ilqr.solved, "solve first"
            solver_ilqr = True
            N = self.ilqr.LP.N
            x = self.ilqr.x
            x0 = self.ilqr.LP.x0
        else:
            N = len(x)
            x0 = x[0]
            solver_ilqr = False

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-7, 7), ylim=(-5, 5))

        ax.grid()

        if solver_ilqr:
            line_goal, = ax.plot([], [], 'ko-', lw=2, label="goal")
            line_goal.set_data(*self.position(self.ilqr.LP.cost.goal))

        line_init, = ax.plot([], [], 'mo-', lw=2, label="initial")
        line_init.set_data(*self.position(x0))

        line, = ax.plot([], [], 'ro-', lw=2)

        ax.legend()
        def init():
            line.set_data([], [])
            return line,

        def animate(t):
            line.set_data(*self.position(x[t]))
            return line,

        animate(0)

        ani = animation.FuncAnimation(fig, animate, frames=N, init_func=init)
        ani.save(name+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])


            
