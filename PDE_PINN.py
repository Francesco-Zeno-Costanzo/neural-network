import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib as mp
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(69420)
np.random.seed(69420)

class Layer(nn.Module):
    ''' Class for NN layers
    '''
    def __init__(self, n_in, n_out, act):
        '''
        Creation of the layers

        Parameters
        ----------
        n_in : int
            number of neurons of the previous layer
        n_out : int
            number og neurons of the current layer
        act : torch.nn function
            activation functionn of the layer
        '''
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.act   = act

    def forward(self, x):
        '''
        Feedforward through the single layer

        Parameter
        ---------
        x : torch.tensor
            input of the layer
        '''
        x = self.layer(x)
        
        if self.act:
            x = self.act(x)
        return x

class NN(nn.Module):
    """
    Class for the neural network
    """
    def __init__(self, dim_in, dim_out, layers, r_max, r_min, act=nn.Sigmoid()):
        '''

        Parameters
        ----------
        dim_in : int
            dimension of the input
        dim_out : int
            dimension of the output
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
        '''
        super().__init__()
        self.net = nn.ModuleList()

        layers = layers + [layers[-1]]   # To obtain the exact number of hidden layes 

        self.net.append(Layer(dim_in, layers[0], act))            # Input layer
        for i in range(1, len(layers)):                           # Hidden layer
            self.net.append(Layer(layers[i-1], layers[i], act))
        self.net.append(Layer(layers[-1], dim_out, act=None))     # Output layer

        self.r_max = torch.tensor(r_max, dtype=torch.float).to(device)
        self.r_min = torch.tensor(r_min, dtype=torch.float).to(device)
        
        self.net.apply(self.init_weights)

    def init_weights(self, l):
        if isinstance(l, nn.Linear):
            torch.nn.init.xavier_uniform_(l.weight.data)
            torch.nn.init.zeros_(l.bias.data)

    def forward(self, x):
        '''
        Feedforward through the all network

        Parameter
        ---------
        x : torch.tensor
            input of the layer
        '''

        out = (x - self.r_min) / (self.r_max - self.r_min)  # Min-max scaling
        
        for layer in self.net: # loop over all layers
            out = layer(out)
        return out


class PINN:
    '''
    Physics informed neural network
    '''

    def __init__(self, layers, r_max, r_min):
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
        '''
        self.net = NN(dim_in=2, dim_out=1, layers=layers, r_max=r_max, r_min=r_min).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def f(self, xt):
        ''' Pde we want to solve id the form f(x, t) = 0
        '''
        xt = xt.clone()
        xt.requires_grad = True

        u    = self.net(xt)                                                     # solution
        u_xt = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]           # du both along x and t
        u_x  = u_xt[:, 0]                                                       # du/dx
        u_t  = u_xt[:, 1]                                                       # du/dt
        u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0]   # d^2u/dx^2
        PDE = u_t - u_xx*0.5
        return PDE

    def train(self, n_epoch, xt_0, u_0, xt_bc, u_bc, xt_f):
        '''
        Train of the nework

        n_epoch : int 
            number of ecpoch of train
        xt_0 : torch.tensor
            point of initial condition
        u_0 : torch.tensor
            initial condition for out problem
        xt_bc : torch.tensor
            point of boundary condition
        u_bc : torch.tensor
            value of the function at the boundary
        xt_f : torch.tensor
            collocation point, point for pde evaluation
        '''
        Loss = []
        for epoch in range(n_epoch):

            self.optimizer.zero_grad() # to make the gradients zero

            # Loss from initial condition
            u0_pred = self.net(xt_0)
            mse_0   = torch.mean(torch.square(u0_pred - u_0))

            # Loss from boundary condition
            u_bc_pred = self.net(xt_bc)
            mse_bc    = torch.mean(torch.square(u_bc_pred - u_bc))

            # Loss from PDE 
            f_pred = self.f(xt_f)
            mse_f  = torch.mean(torch.square(f_pred))

            loss = mse_0 + mse_bc + mse_f
            
            loss.backward()
            self.optimizer.step()

            with torch.autograd.no_grad():
                Loss.append(loss.data)
                if epoch % 100 == 0:
                    print(f"epoch: {epoch} t_0: {mse_0.data:.3e} bc: {mse_bc.data:.3e} pde: {mse_f.data:.3e}")
        
        return Loss

start = time.time()

#=======================================================
# Computational parameters
#=======================================================

# Interval size
x_min = 0.0
x_max = 2.0
t_min = 0.0
t_max = 1.0
# Number of points
N_x   = 200
N_t   = 100
N_col = 200

# Set initial condition u(x, t=0) = f(x) 
xt_0 = np.random.uniform([x_min, 0], [x_max, 0], size=(N_x, 2))
#u_0  = np.ones_like(xt_0[:, 0:1]) * 2
u_0 = 2*np.exp(-((xt_0 - 1)/0.1)**2)

# Set boundary Condition
# u(0, t) = 0 & u(L, t) = 0 for all t > 0

# Left side
xt_bc_0 = np.random.uniform([x_min, t_min], [x_min, t_max], size=(N_t // 2, 2))
u_bc_0  = np.zeros((len(xt_bc_0), 1))
#u_bc_0  = np.ones((len(xt_bc_0), 1))*4

# Right side
xt_bc_1 = np.random.uniform([x_max, t_min], [x_max, t_max], size=(N_t // 2, 2))
u_bc_1  = np.zeros((len(xt_bc_1), 1))

# All boundary condition
xt_bc = np.vstack([xt_bc_0, xt_bc_1])
u_bc  = np.vstack([u_bc_0,  u_bc_1])

# Collocation points
xt_f = np.random.uniform([x_min, t_min], [x_max, t_max], (N_col, 2))
xt_f = np.vstack([xt_0, xt_bc, xt_f])

#=======================================================
# Convert to Tensor
#=======================================================

xt_0  = torch.tensor(xt_0,  dtype=torch.float).to(device)
u_0   = torch.tensor(u_0,   dtype=torch.float).to(device)

xt_bc = torch.tensor(xt_bc, dtype=torch.float).to(device)
u_bc  = torch.tensor(u_bc,  dtype=torch.float).to(device)

xt_f  = torch.tensor(xt_f,  dtype=torch.float).to(device)

#=======================================================
# Creation of network and train
#=======================================================

n_epoch = 5000 + 1
pinn = PINN([20, 20, 20, 20], [x_min, t_min], [x_max, t_max])
Loss = pinn.train(n_epoch, xt_0, u_0, xt_bc, u_bc, xt_f)

end = time.time() - start
print(f"Elapsed time {end}")

#=======================================================
# Plot
#=======================================================

plt.figure(0)
plt.plot(range(n_epoch), Loss)

fig = plt.figure(1)
ax  = fig.add_subplot(projection='3d')

x = np.arange(x_min, x_max, 0.01)
t = np.arange(t_min, t_max, 0.01)

X, T = np.meshgrid(x, t)

x = X.reshape(-1, 1)  # Reshape points in the same format
t = T.reshape(-1, 1)  # for the input of the network

xt_p = np.hstack([x, t])
xt_p = torch.tensor(xt_p, dtype=torch.float).to(device)

u_pred = pinn.net(xt_p)
u_pred = u_pred.detach().cpu().numpy()
U      = u_pred.reshape(X.shape)

ax.plot_surface(X, T, U, cmap=mp.cm.coolwarm, vmax=np.max(U)/2,linewidth=0,rstride=2, cstride=2)
ax.set_title('Heat diffussion')
ax.set_ylabel('Time')
ax.set_xlabel('Distance')
ax.set_zlabel('Temperature')

plt.show()