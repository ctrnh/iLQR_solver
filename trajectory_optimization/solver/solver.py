import numpy as np
import matplotlib.pyplot as plt
class Solver:
    def __init__(self, problem):
        self.LP = problem
        self.solved = False
        self.x = [None for i in range(self.LP.N + 1)] #(x_0,... , x_N-1, x_N)
        self.u = [None for i in range(self.LP.N)] # (u_0, ..., u_N-1)
        self.J = [None for i in range(self.LP.N + 1)] # Cum loss (J_0, ..., J_N)

    def plot_loss(self):
        if self.solved:
            plt.plot(self.J, label="Cum loss")
            plt.title("Loss wrt iterations")
            plt.grid()
            plt.legend()
        else:
            "Solve first"

    def plot_u(self):
        if not self.solved:
            "Solve first"
        else:
            u_dim = self.LP.dyn.u_dim
            for i in range(u_dim):
                if u_dim == 1:
                    u_i = [float(self.u[t]) for t in range(self.LP.N)]
                    plt.plot(u_i, label=f"u")
                else:
                    u_i = [float(self.u[t][i]) for t in range(self.LP.N)]

                    plt.plot(u_i, label=f"u_{i}")
                plt.legend()
            plt.grid()
            plt.title("Control u wrt iterations")

    def plot_x(self, idx=None):
        if not self.solved:
            "Solve first"
        else:
            xf = self.LP.cost.goal
            x_dim = self.LP.dyn.state_dim
            if idx is None:
                idx = np.arange(x_dim)
            for i in range(x_dim):
                if x_dim == 1:
                    plt.plot(self.x, label=f"x")
                    plt.plot(xf*np.ones(self.LP.N+1), "k--", label=f'xf at N')
                else:
                    if i in idx:
                        plt.plot(xf[i]*np.ones(self.LP.N+1), "k--")#, label=f'xf_{i} at N')
                        x_i = [self.x[t][i] for t in range(self.LP.N+1)]
                        plt.plot(x_i, label=f"x_{i}")
                plt.legend()
            plt.grid()
            plt.title(f"x, goal:{self.LP.cost.goal}")
