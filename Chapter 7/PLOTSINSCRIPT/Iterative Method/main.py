from FiniteDifference import solve_waveequation1D
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import math
import copy
import utilities
from IterativeMethod import IterativeInverseSolverWaveEquation1D

import matplotlib
import matplotlib.font_manager
matplotlib.rcParams["figure.dpi"] = 300
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],
             'size' : 16})
rc('text', usetex=True)

device = torch.device('cpu') 
torch.manual_seed(20)

# Model parameters
L = 1 # Length of bar
Nx = 120 # Number of points in spatial dimension
dt = 5e-3 # Time stepping size
N = 300 # Number of time steps

# Neural Network parameters
lr = 1e-2
epochs = 1000
patience = 100
factor_lr = 1e-1
c_weight = 1e0

# Position of sensors
sel=(0,-1) 
nsel = len(sel)

# Source term
f0 = 2e1
t0 = 0.1
def dirac(x):
    x = x*0
    x[:,(30,60,90)] = 1 
    return x
f_exact = lambda x, t : 1e2 * dirac(x)*(f0**2 * (t-t0) * torch.exp(-f0**2*(t-t0)**2))

I = lambda x : x*0
V = lambda x : x*0

# Grid generation TODO create function
x = torch.linspace(0, L, Nx + 1, device=device)
dx = x[1] - x[0]
t = torch.linspace(0, N * dt, N + 1, device=device)
x = x.view(1,-1).repeat(N+1,1)
t = t.view(-1,1).repeat(1, Nx+1)
f = f_exact(x, t)

# wave velocity
cm = x[0]*0 + 1.
cm[30:40] = cm[30:40]*0.4

#cm[50:90] = cm[50:90]*0.7


# measurement
um, tt, cpu, T = solve_waveequation1D(I, V, f, cm, x[0], Nx, dt, N, bc='Neumann')
uin = um[:,sel].view(1,-1,N+1) #TUPLE AS INPUT

        
        
Test = IterativeInverseSolverWaveEquation1D(uin,sel,N,Nx,x,I,V,f,dt)
Test.buildModel()
history_cpred, history_cost = Test.train(epochs,lr)

c_pred = Test.predict_c(x[0])
u_pred, tt, cpu, T = solve_waveequation1D(I, V, f,
                                          c_pred, x[0],
                                          Nx, dt, N, 
                                          bc='Neumann')

utilities.plot_history(history_cost, filename='history')

utilities.plot_field(x, t, u_pred.cpu().detach(), N+1, Nx+1, title='Prediction of $u$', filename='predictionu')
utilities.plot_field(x, t, um, N+1, Nx+1, title='Solution of $u$', filename='solutionu')
utilities.plot_field(x, t, abs(u_pred.cpu().detach()-um), N+1, Nx+1, title='Absolute Error of $u$', filename='erroru')

fig, ax = plt.subplots()
ax.plot(x[0].detach().numpy(), c_pred.detach().numpy(),color='k',label='$v_{p_{\\textrm{pred}}}$')
ax.plot(x[0].detach().numpy(), cm.detach().numpy(),color='r',linestyle='--',label='$v_{p_{\\textrm{analytic}}}$')
ax.legend()
ax.set_title('Wave Velocity')
ax.set_xlabel('$x$')
ax.set_ylabel('$v_{p}$')
#ax.grid()
fig.tight_layout()
plt.show()
plt.savefig('IterativeFD.eps')

        
# # animation learning history
# from matplotlib import animation

# # animation of c
# fig, ax = plt.subplots()
# mat, = ax.plot([], [], color='k', linestyle='-', label='Prediction')
# text = ax.set_title(0)

# xani_c =x[0].cpu().detach().numpy()

# def animate_c(i):
#     y = history_cpred[i]
#     mat.set_data(xani_c, y)
#     text.set_text('Wave Velocity - Neural Network - ' + str(i*10) + ' Epochs')
#     return mat, text

# ax.plot(xani_c, cm.cpu().detach().numpy(),color='r',linestyle='--',label='Solution')
# ax.legend(loc='lower right')
# ax.axis([min(xani_c),max(xani_c),np.amin(history_cpred),np.amax(history_cpred)])
# ax.set_xlabel('$x$')
# ax.set_ylabel('$v_{p}$')
# ax.grid()
# #ax.plot(0.1,0, marker="x", markersize=10)
# #ax.plot(0.6,0, marker="x", markersize=10)
# ani = animation.FuncAnimation(fig, animate_c, blit=False, interval=200, frames =(epochs // 10)+1)
# fig.tight_layout()
# plt.show()
# writer = animation.PillowWriter(fps=5)
# ani.save('training_history_c.gif', writer=writer)



        
        