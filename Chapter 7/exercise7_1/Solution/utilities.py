import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

device = torch.device('cpu')

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
    
def plot_field(x, t, z, tsize, xsize, title, filename=None):
    """Plot a 2D field."""
    x = x.reshape(tsize,xsize).cpu().detach().numpy()
    t = t.reshape(tsize,xsize).cpu().detach().numpy()
    z = z.reshape(tsize,xsize).cpu().detach().numpy()
    
    # Set up plot
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    ax.axis('on')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')

    # Plot the field
    cp = ax.contourf(x, t, z, levels=12, cmap=plt.cm.jet)

    # Add a colorbar to the plot
    fig.colorbar(cp)
    
    fig.tight_layout()
    plt.show()
    if filename is not None:
        plt.savefig(filename + ".eps")
        
def plot_wavevelocity(x, c_pred, c_solution, filename=None):
    """Plot the wave velocity."""
    
    # Set up plot
    fig, ax = plt.subplots()
    ax.set_title('Wave Velocity')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$v_{p}$')
    
    # Plot the wave velocity
    ax.plot(x.detach().numpy(), c_pred.detach().numpy(),color='k',label='$v_{p_{pred}}$')
    ax.plot(x.detach().numpy(), c_solution.detach().numpy(),color='r',linestyle='--',label='$v_{p_{analytic}}$')
    
    ax.legend()
    fig.tight_layout()
    plt.show()
    if filename is not None:
        plt.savefig(filename+".eps")
        
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
        

        
def animate_history(x, history_cpred, c_solution, epochs, filename=None):
    """Animate the learning history of the wave velocity."""
    
    # Set upt plot
    fig, ax = plt.subplots()
    mat, = ax.plot([], [], color='k', linestyle='-', label='Prediction')
    text = ax.set_title(0)
    ax.axis([min(x),max(x),np.amin(history_cpred),np.amax(history_cpred)])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$v_{p}$')

    def animate_c(i):
        """Helper function for the animation."""
        y = history_cpred[i]
        mat.set_data(x, y)
        text.set_text('Wave Velocity - Neural Network - ' + str(i*10) + ' Epochs')
        return mat, text

    # Plot the solution
    ax.plot(x, c_solution.cpu().detach().numpy(),color='r',linestyle='--',label='Solution')

    # Animate the learned solution at each given epoch
    ani = animation.FuncAnimation(fig, animate_c, blit=False, interval=200, frames=(epochs // 10))
    
    ax.legend(loc='lower right')
    fig.tight_layout()
    plt.show()
    
    if filename is not None:
        writer = animation.PillowWriter(fps=5)
        ani.save(filename+".gif", writer=writer)

        