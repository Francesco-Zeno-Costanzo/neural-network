"""
Code for solving Poisson's equation with a neural network
"""
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib as mp
import matplotlib.pyplot as plt

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

    def f(self, xy):
        ''' Pde we want to solve id the form f(x, y) = 0
        '''
        xy = xy.clone()
        xy.requires_grad = True

        u    = self.net(xy)                                                     # solution
        u_xy = torch.autograd.grad(u.sum(),   xy, create_graph=True)[0]         # du both along x and y
        u_x  = u_xy[:, 0]                                                       # du/dx
        u_y  = u_xy[:, 1]                                                       # du/dy
        u_xx = torch.autograd.grad(u_x, xy, 
                                   grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0][:, 0]                  # d^2u/dx^2
        u_yy = torch.autograd.grad(u_y, xy,
                                   grad_outputs=torch.ones_like(u_y),
                                   create_graph=True)[0][:, 1]                  # d^2u/dy^2

        PDE = u_yy + u_xx + torch.sin(xy[:, 0]*np.pi)*torch.sin(xy[:,1]*np.pi)
        
        return PDE

    def train(self, n_epoch, domain_bc, u_bc, domain_f):
        '''
        Train of the nework

        Parameters
        ----------
        n_epoch : int 
            number of ecpoch of train
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

            # Loss from boundary condition
            u_bc_pred = self.net(domain_bc)
            mse_bc    = torch.mean(torch.square(u_bc_pred - u_bc))

            # Loss from PDE 
            f_pred = self.f(domain_f)
            mse_f  = torch.mean(torch.square(f_pred))

            loss = mse_bc + mse_f
            
            loss.backward()
            self.optimizer.step()

            with torch.autograd.no_grad():
                Loss.append(loss.data.detach().cpu().numpy())
                if epoch % 100 == 0:
                    print(f"epoch: {epoch} bc: {mse_bc.data:.3e} pde: {mse_f.data:.3e}")
        
        return Loss

start = time.time()

#=======================================================
# Computational parameters
#=======================================================

# Interval size
x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 1.0
# Number of points
N_x   = 200
N_y   = 200
N_col = 400

# Set boundary Condition
# u(x_min, y) = 0 & u(x_max, y) = 0
# u(x, y_min) = 0 & u(x, y_max) = 0

# Bot side
xy_bc_1 = np.random.uniform([x_min, y_min], [x_max, y_min], size=(N_x // 2, 2))
u_bc_1  = np.zeros((len(xy_bc_1), 1))

# Top side
xy_bc_2 = np.random.uniform([x_min, y_max], [x_max, y_max], size=(N_x // 2, 2))
u_bc_2  = np.zeros((len(xy_bc_2), 1))

# Left side
xy_bc_3 = np.random.uniform([x_min, y_min], [x_min, y_max], size=(N_y // 2, 2))
u_bc_3  = np.zeros((len(xy_bc_3), 1))

# Right side
xy_bc_4 = np.random.uniform([x_max, y_min], [x_max, y_max], size=(N_y // 2, 2))
u_bc_4  = np.zeros((len(xy_bc_4), 1))

# All boundary condition
domain_bc = np.vstack([xy_bc_1, xy_bc_2, xy_bc_3, xy_bc_4])
u_bc      = np.vstack([u_bc_1,  u_bc_2,  u_bc_3,  u_bc_4])

# Collocation points
xy_f = np.random.uniform([x_min, y_min], [x_max, y_max], (N_col, 2))
domain_f = np.vstack([domain_bc, xy_f])

#=======================================================
# Convert to Tensor
#=======================================================

domain_bc = torch.tensor(domain_bc, dtype=torch.float).to(device)
u_bc  = torch.tensor(u_bc,  dtype=torch.float).to(device)

domain_f  = torch.tensor(domain_f,  dtype=torch.float).to(device)

#=======================================================
# Creation of network and train
#=======================================================

n_epoch = 5000 + 1
pinn = PINN([20, 20, 20, 20], [x_min, y_min], [x_max, y_max])
Loss = pinn.train(n_epoch, domain_bc, u_bc, domain_f)

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
y = np.arange(y_min, y_max, 0.01)

X, Y = np.meshgrid(x, y)

# Analytical solution
sol  = (np.sin(np.pi * X) * np.sin(np.pi * Y)) / (2*np.pi**2)

x = X.reshape(-1, 1)  # Reshape points in the same format
y = Y.reshape(-1, 1)  # for the input of the network

domain_p = np.hstack([x, y])
domain_p = torch.tensor(domain_p, dtype=torch.float).to(device)

u_pred = pinn.net(domain_p)
u_pred = u_pred.detach().cpu().numpy()
U      = u_pred.reshape(X.shape)

ax.plot_surface(X, Y, U, cmap=mp.cm.plasma, vmax=np.max(U)/2,linewidth=0,rstride=2, cstride=2)
ax.set_title("Poisson equation")
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.set_zlabel('u(x, y)')


plt.figure(2)
plt.title("Error")
plt.xlabel('x')
plt.ylabel('y')
error  = abs(U - sol)
levels = np.linspace(np.min(error), np.max(error), 40)
c=plt.contourf(X, Y, error, levels=levels, cmap='plasma')
plt.colorbar(c)

plt.show()