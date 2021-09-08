import torch
import matplotlib.pyplot as plt
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generateGrid2d(L, Nx, N, dt, device):
    """Generate an evenly spaced grid in the spatio-temporal domain."""
    
    # Generate the spatial grid
    x = torch.linspace(0, L, Nx + 1, device=device)
    dx = x[1] - x[0]
    
    # Generate the temporal grid
    t = torch.linspace(0, N * dt, N + 1, device=device)
    
    # Reshape to 2D grid
    x = x.view(1,-1).repeat(N+1,1)
    t = t.view(-1,1).repeat(1, Nx+1)
    return x, t
    
def bump(x,a,b,c):
    """Bump function for the definition of a differentiable wave velocity"""
    return np.where((x<(b+c)) & (x>(-b+c)),1-a*torch.exp(-1/(1-((x-c)/b)**2)),1)

def plot_history(history_cost=[], history_cost_=[], filename=None):
    """Plot training history."""
    
    # Set up plot
    fig, ax = plt.subplots()
    plt.yscale('log')
    
    if len(history_cost) != 0:
        epochs = len(history_cost)
        ax.plot(np.linspace(1,epochs,epochs), history_cost, color='k', label='Training Cost')
    if len(history_cost_) != 0:
        epochs = len(history_cost_)
        ax.plot(np.linspace(1,epochs,epochs), history_cost_, color='r', linestyle='--', label='testing cost')
    
    ax.legend()
    ax.set_title('Cost function history')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('$L_{2}$-error')
    
    fig.tight_layout()
    plt.show()
    if filename is not None:
        plt.savefig(filename + ".eps")
        
def generate_c(nx, nt, nc, x): 
    """Generate nc wave velocity profiles."""
    
    # Arrays to generate the wave velocity with its corresponding parametrization
    c_train = torch.ones((nc, nt, nx), device=device)
    par_train = torch.ones((nc, 3), device=device)
    
    # Loop over number of wave velocity profiles
    for i in range(nc):
        # Generate parameters of the parametrization
        a = random.uniform(0,np.exp(1))
        b = random.uniform(0,0.5)
        c = random.uniform(0, 1)
        
        c_train[i, :] = torch.from_numpy(bump(x,a,b,c))
        
        #c_train[i, :, a:b] = c_train[i, :, a:b] * 0 + c
        par_train[i, 0] = a
        par_train[i, 1] = b
        par_train[i, 2] = c
        
    c_train = c_train.view(nc, 1, nt, nx)
    return c_train, par_train
     
def plot_wavevelocities(P_pred, P_solution, A, shift, title, filename=None):
    """Plot example wave velocity predictions with solutions from a data set."""

    # Set up plot
    fig, ax = plt.subplots(A,A)
    
    x = torch.linspace(0,1,120, device='cpu')
    
    # Loop over all subplots
    for i in range(A):
        for j in range(A):    
            n = A*j+i+shift
            
            a = P_pred[n].cpu().detach().numpy()[0]
            b = P_pred[n].cpu().detach().numpy()[1]
            c = P_pred[n].cpu().detach().numpy()[2]
           
            c_pred = bump(x,a,b,c)
            
            a = P_solution[n].cpu().detach().numpy()[0]
            b = P_solution[n].cpu().detach().numpy()[1]
            c = P_solution[n].cpu().detach().numpy()[2]
            
            c_test = bump(x,a,b,c)
            
            if i==0 and j==0:
                ax[i,j].plot(x.cpu().detach().numpy(),c_pred, color='k', label='Prediction')
                ax[i,j].plot(x.cpu().detach().numpy(),c_test, color='r', linestyle='--', label='Solution')
            else:
                ax[i,j].plot(x.cpu().detach().numpy(),c_pred, color='k')
                ax[i,j].plot(x.cpu().detach().numpy(),c_test, color='r', linestyle='--')
            ax[i,j].set_xlabel('$x$')
            ax[i,j].set_ylabel('$v_{p}$')
            
    fig.tight_layout() 
    fig.subplots_adjust(top=0.85)
    fig.suptitle(title,y=0.98)
    fig.legend(bbox_to_anchor=(1.015,1.02))
    plt.show()
    if filename is not None:
        plt.savefig(filename+".eps")







