import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib.font_manager
matplotlib.rcParams["figure.dpi"] = 300
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],
             'size' : 16})
rc('text', usetex=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generateGrid(x_size, t_size, xmax=1, tmax=1, double=True):
    if double==True:
        dtype=torch.float64
    else:
        dtype=torch.float32
    x = torch.linspace(0,1,x_size, device=device, dtype=dtype)
    t = torch.linspace(0,1,t_size, device=device, dtype=dtype)
    x = x.view(1,1,1,-1)
    t = t.view(1,1,-1,1)
    x = x.repeat(1,1,t_size,1)
    t = t.repeat(1,1,1,x_size)
    return x, t

def grad_h_2(u,imsize, order=1): #for x
    #center
    dx = 1/(imsize-1)
    if order==1:
        dduc = u[:,:,:,0:-2] - 2* u[:,:,:,1:-1] + u[:,:,:,2::] 
        dduc = dduc / dx ** 2 
        dduc = dduc[:,:,1:-1,:]
    if order==2:
        dduc = (-u[:,:,:,4::] + 16*u[:,:,:,3:-1] - 30*u[:,:,:,2:-2]
                + 16*u[:,:,:,1:-3] - u[:,:,:,0:-4])
        dduc = dduc/(12*dx**2)
        dduc = dduc[:,:,2:-2,:]
    
    return dduc

def grad_v_2(u,imsize, order=1): #for t
    #center 
    dt = 1/(imsize-1)
    if order==1:
        dduc = u[:,:,0:-2,:] - 2* u[:,:,1:-1,:] + u[:,:,2::,:]
        dduc = dduc / dt ** 2
        dduc = dduc[:,:,:,1:-1]
    if order==2:
        dduc = (-u[:,:,4::,:] + 16*u[:,:,3:-1,:] - 30*u[:,:,2:-2,:]
                + 16*u[:,:,1:-3,:] - u[:,:,0:-4,:])
        dduc = dduc/(12*dt**2)
        dduc = dduc[:,:,:,2:-2]
        
    return dduc
    
def plot_field(x, t, z, tsize, xsize, title, filename=None):
    x = x.reshape(tsize,xsize).cpu().detach().numpy()
    t = t.reshape(tsize,xsize).cpu().detach().numpy()
    z = z.reshape(tsize,xsize).cpu().detach().numpy()
    
    if plt.get_backend() == 'agg':
        plt.switch_backend('Qt5Agg')
    
    # Set up plot
    fig, ax = plt.subplots(1, 1)
    #plt.gca().set_aspect('equal', adjustable='box')
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
        
def plot_history(history_cost=[], history_cost_=[], filename=None):
    if plt.get_backend() == 'agg':
        plt.switch_backend('Qt5Agg')
        
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
    #ax.grid()
    fig.tight_layout()
    plt.show()
    if filename is not None:
        plt.savefig(filename + ".eps")
        
def generate_c(nx, nc, sel):
    np.random.seed(1)
    x = np.linspace(0, 1 , nx, dtype=np.float)
    
    c_train=np.zeros((nc,nx,nx))
    
    func_space = np.array([1.+0*x, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8])
    #TODO improve
    
    
    for i in range(nc):
        shift = np.random.rand(len(func_space))
        func_space = np.array([1.+0*(x-shift[0]), 
                               (x-shift[1]),
                               (x-shift[2])**2,
                               (x-shift[3])**3,
                               (x-shift[4])**4,
                               (x-shift[5])**5,
                               (x-shift[6])**6,
                               (x-shift[7])**7, 
                               (x-shift[8])**8])
    
        select = np.random.rand(len(func_space),1)*10.
        select[:(len(func_space)-sel)]=0.
        np.random.shuffle(select)
        temp = np.sum(func_space*select,0)
        temp = np.reshape(temp,(1,nx))
        temp = np.repeat(temp,nx,axis=0)
        c_train[i] = temp
    
    #convert to torch tensor
    c_train = torch.from_numpy(c_train) 
    c_train.requires_grad = True
    c_train = c_train.to(device)
    c_train = c_train.view(nc,1,nx,nx)
    c_train = torch.as_tensor(c_train, dtype = torch.float)
    
    return c_train
