"""
Example: Cartpole
Dynamics :
\begin{align*}
\ddot x &= \frac{F + m\sin \theta (g \cos \theta + \ell \dot \theta^2)}{M + m \sin^2 \theta} \\
\ddot \theta &= \frac{-(M+m)g \sin \theta - m \ell \dot \theta^2 \cos \theta \sin \theta - F \cos \theta}{\ell (M + m \sin^2 \theta)}
\end{align*}
"""
class CartPoleEnv(Environment):
    def __init__(self, dt, m=1, M=1, l=1):
        super().__init__(dt,
                         state_dim=4,
                         u_dim=1)

        self.dynamics = CartPoleDynamics(dt=dt, m=m, M=M, l=l)

    def position(self,x):
        """
        Coordinates of cart and pole given a state
        """
        cart_x = [x[0][0], x[0][0] + self.dynamics.l * np.sin((x[1][0]))]
        pole_y = [0, -self.dynamics.l * np.cos((x[1][0]))]
        return (cart_x, pole_y)



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
                   rho_reg=0
                  ):
        """
        solve with iLQR:
        - initialize with random uniform * 1e-3
        - Q, R = eye
        """
        if R is None:
            R = np.eye(self.u_dim)

        if Q is None:
            Q = np.eye(self.state_dim)

        cost = QuadraticCost(dt=self.dt,
                    N=N,
                    goal=xf,
                    Qf=Qf,
                    Q=Q,
                    R=R)

        print("Initial cost: x0, u= 0, J = ", cost.loss(x=x0,u=np.zeros((self.u_dim,1)),k=0))
        print("Last cost: x0, u= 0, J = ", cost.loss(x=x0,u=np.zeros((self.u_dim,1)),k=N))

        problem = Problem(dynamics=self.dynamics,
                                 cost=cost,
                                 dt=self.dt,
                                 x0=x0
                                )

        self.ilqr = iLQR(problem=problem,
                           n_traj=n_traj,
                           u_init=[(1e-6)*np.random.uniform(size=(self.u_dim,1)) for i in range(N)],
                        alpha_init=alpha_init,
                        crit_alpha=crit_alpha,
                        rho_reg=rho_reg)
        self.ilqr.solve()


if __name__ == '__main__':
    N = 100
    dt = 0.05

    #qf=1
    m = 0.2
    M = 1
    l = .5

    CP_env = CartPoleEnv(dt=dt,
                          m=m,
                          M=M,
                          l=l)


    x0 = np.array([[0],[0.*np.pi], [0], [0]])

    xf =  np.array([[-4], [1.0*np.pi], [0], [0]])


    Q = 1e-2*np.eye(4)
    #Q[1][1] = 10
    #Q[0][0] = 10

    Qf = 100*np.eye(4)
    #Qf[0][0] = 11

    R = 1e-1*np.eye(1)
    CP_env.solve_iLQR(x0=x0,
                      xf=xf,
                      N=N,
                      n_traj=70,
                     alpha_init=1,
                     crit_alpha=1,
                     Q=Q,
                     Qf=Qf)

    plt.figure(figsize=(15,3))
    plt.subplot(131)
    CP_env.ilqr.plot_loss()
    plt.subplot(132)
    CP_env.ilqr.plot_u()
    plt.subplot(133)
    CP_env.ilqr.plot_x()
    plt.show()
    CP_env.create_animation(name=f"swingup_move2_Q-{np.diag(Q)}_N_{N}_dt_{dt}")
