import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import matplotlib.pyplot as plt
import numpy as np
import random

import matplotlib
import matplotlib.font_manager
matplotlib.rcParams["figure.dpi"] = 300
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],
             'size' : 16})
rc('text', usetex=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu') #OBS GPU DEACTIVATED

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
    x = x.reshape(tsize,xsize) #.cpu().detach().numpy()
    t = t.reshape(tsize,xsize) #.cpu().detach().numpy()
    z = z.reshape(tsize,xsize) #.cpu().detach().numpy()
    
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
        plt.savefig(filename + ".png")
        
def plot_history(history_cost=[], history_cost_=[], filename=None):
    if plt.get_backend() == 'agg':
        plt.switch_backend('Qt5Agg')
        
    fig, ax = plt.subplots()
    plt.yscale('log')
    #plt.grid(which='both')
    if len(history_cost) != 0:
        epochs = len(history_cost)
        ax.plot(np.linspace(1,epochs,epochs), history_cost, color='k', label='Training Cost')
    if len(history_cost_) != 0:
        epochs = len(history_cost_)
        ax.plot(np.linspace(1,epochs,epochs), history_cost_, color='r', linestyle='--', label='Testing Cost')
    ax.legend()
    ax.set_title('Cost function history')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost function $C$')
    fig.tight_layout()
    plt.show()
    if filename is not None:
        plt.savefig(filename + ".eps")
        plt.savefig(filename + ".png")
        
def generate_c(nx, nt, nc): #TODO MAKE MORE ARBITRARY  
    c_train = torch.ones((nc, nt, nx), device=device)
    par_train = torch.ones((nc, 3), device=device)
    
    for i in range(nc):
        a = random.randint(1,nx-3)
        b = random.randint(a+2,nx-1)
        c = random.uniform(0, 1)
        c_train[i, :, a:b] = c_train[i, :, a:b] * 0 + c
        par_train[i, 0] = a
        par_train[i, 1] = b
        par_train[i, 2] = c
    c_train = c_train.view(nc, 1, nt, nx)
    return c_train, par_train
