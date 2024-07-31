"""
Code for solving wave equation with a neural network
"""
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torchneural import NN


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(69420)
np.random.seed(69420)


class PINN:
    '''
    Physics informed neural network
    '''

    def __init__(self, layers, r_max, r_min, act=nn.Tanh()):
        '''
        Parameters
        ----------
        layers : list
            list which must contain the number of neurons for each layer
            the number of layers is len(layers) and layers[i] is the 
            number of neurons on the i-th layer. Only hidden layers must
            be declared
        r_max : torch.tensor
            max value of the input parameters
            e.g. if we are in the square  0<x<1 0<y<1 r_max = [1, 1] 
        r_min : torch.tensor
            min value of the input parameters
            e.g. if we are in the square  0<x<1 0<y<1 r_max = [0, 0]
        act : torch.. function, optional, default torch.nn.Sigmoid
            activation functionn of the layer
        '''
        self.net = NN(dim_in=2, dim_out=1, layers=layers, r_max=r_max, r_min=r_min, act=act).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def f(self, xt):
        ''' Pde we want to solve id the form f(x, t) = 0
        '''
        xt = xt.clone()
        xt.requires_grad = True

        u    = self.net(xt)                                                     # solution
        u_xt = torch.autograd.grad(u.sum(),   xt, create_graph=True)[0]         # du both along x and t
        u_x  = u_xt[:, 0]                                                       # du/dx
        u_t  = u_xt[:, 1]                                                       # du/dt
        u_xx = torch.autograd.grad(u_x, xt,
                                   grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0][:, 0]                  # d^2u/dx^2
        u_tt = torch.autograd.grad(u_t, xt,
                                   grad_outputs=torch.ones_like(u_t),
                                   create_graph=True)[0][:, 1]                  # d^2u/dt^2  


        PDE = u_tt - u_xx

        return PDE

    def train(self, n_epoch, xt_0, u_0, domain_bc, u_bc, domain_f):
        '''
        Train of the nework

        Parameters
        ----------
        n_epoch : int 
            number of ecpoch of train
        xt_0 : torch.tensor
            point of initial condition
        u_0 : torch.tensor
            initial condition
        domain_bc : torch.tensor
            point of boundary condition
        u_bc : torch.tensor
            value of the function at the boundary
        domain_f : torch.tensor
            collocation point, point for pde evaluation
        
        Return
        ------
        Loss : list
            training loss
        '''
        Loss = []
        for epoch in range(n_epoch):

            self.optimizer.zero_grad() # to make the gradients zero

            # Loss from initial condition
            u0_pred = self.net(xt_0)
            mse_0   = torch.mean(torch.square(u0_pred - u_0))

            # Loss from temporal derivative on initial condition
            xt = xt_0.clone()
            xt.requires_grad = True
            u    = self.net(xt)
            u_xt = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
            u_t  = u_xt[:, 1]
            mse_dudt = torch.mean(torch.square(u_t))

            # Loss from boundary condition
            u_bc_pred = self.net(domain_bc)
            mse_bc    = torch.mean(torch.square(u_bc_pred - u_bc))

            # Loss from PDE 
            f_pred = self.f(domain_f)
            mse_f  = torch.mean(torch.square(f_pred))

            loss = mse_0 + mse_bc + mse_f + mse_dudt
            
            loss.backward()
            self.optimizer.step()

            with torch.autograd.no_grad():
                Loss.append(loss.data.detach().cpu().numpy())
                if epoch % 100 == 0:
                    print(f"epoch: {epoch} t_0: {mse_0.data:.3e} bc: {mse_bc.data:.3e} pde: {mse_f.data:.3e} dudt {mse_dudt.data:.3e}")
        
        return Loss

start = time.time()

#=======================================================
# Computational parameters
#=======================================================

# Interval size
x_min = 0.0
x_max = 1.0
t_min = 0.0
t_max = 1.0
# Number of points
N_x   = 1000
N_t   = 1000
N_col = 1000

# Set initial condition u(x, t=0) = f(x) 
xt_0 = np.random.uniform([x_min, 0], [x_max, 0], size=(N_x, 2))
u_0  = np.sin(2*np.pi*xt_0[:, 0:1])

# Set boundary Condition
# u(0, t) = 0 & u(L, t) = 0 for all t > 0

# Left side
xt_bc_1 = np.random.uniform([x_min, t_min], [x_min, t_max], size=(N_t // 2, 2))
u_bc_1  = np.zeros((len(xt_bc_1), 1))

# Right side
xt_bc_2 = np.random.uniform([x_max, t_min], [x_max, t_max], size=(N_t // 2, 2))
u_bc_2  = np.zeros((len(xt_bc_2), 1))

# All boundary condition
domain_bc = np.vstack([xt_bc_1, xt_bc_2])
u_bc  = np.vstack([u_bc_1,  u_bc_2])

# Collocation points
xt_f = np.random.uniform([x_min, t_min], [x_max, t_max], (N_col, 2))
domain_f = np.vstack([xt_0, domain_bc, xt_f])

#=======================================================
# Convert to Tensor
#=======================================================

xt_0  = torch.tensor(xt_0,  dtype=torch.float).to(device)
u_0   = torch.tensor(u_0,   dtype=torch.float).to(device)

domain_bc = torch.tensor(domain_bc, dtype=torch.float).to(device)
u_bc      = torch.tensor(u_bc,      dtype=torch.float).to(device)

domain_f  = torch.tensor(domain_f,  dtype=torch.float).to(device)

#=======================================================
# Creation of network and train
#=======================================================

n_epoch = 6074 + 1
pinn = PINN([30, 30], [x_min, t_min], [x_max, t_max])
Loss = pinn.train(n_epoch, xt_0, u_0, domain_bc, u_bc, domain_f)

end = time.time() - start
print(f"Elapsed time {end}")

#=======================================================
# Plot
#=======================================================

plt.figure(0)
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.grid()
plt.yscale("log")
plt.plot(range(n_epoch), Loss)


fig = plt.figure(1)
ax  = fig.add_subplot(projection='3d')

x = np.arange(x_min, x_max, 0.01)
t = np.arange(t_min, t_max, 0.01)

X, T = np.meshgrid(x, t)

# Analytical solution
sol  = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * T)

x = X.reshape(-1, 1)  # Reshape points in the same format
t = T.reshape(-1, 1)  # for the input of the network

domain_p = np.hstack([x, t])
domain_p = torch.tensor(domain_p, dtype=torch.float).to(device)

u_pred = pinn.net(domain_p)
u_pred = u_pred.detach().cpu().numpy()
U      = u_pred.reshape(X.shape)

ax.plot_surface(X, T, U)
ax.set_title("Wave equation")
ax.set_ylabel('t')
ax.set_xlabel('x')
ax.set_zlabel('u(x, t)')


plt.figure(2)
plt.title("Error")
plt.xlabel('x')
plt.ylabel('t')
error  = abs(U - sol)
levels = np.linspace(np.min(error), np.max(error), 40)
c=plt.contourf(X, T, error, levels=levels, cmap='plasma')
plt.colorbar(c)


fig = plt.figure(3)
ax = fig.add_subplot()

line1, = ax.plot([], [], 'r', label='Analytical')
line2, = ax.plot([], [], 'b', label='Prediction')
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1.1, 1.1)
ax.grid()
ax.legend()

t = np.arange(t_min, t_max, 0.01)
x = np.arange(x_min, x_max, 0.01)

def update(i):
    line1.set_data(x, sol[i, :])
    line2.set_data(x, U[i, :])
    ax.set_title(f"Time: {t[i]:.2f}")
    return line1, line2

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, len(t), 1), interval=50)

plt.show()