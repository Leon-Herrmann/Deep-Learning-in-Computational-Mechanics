import torch
import numpy as np
from FiniteDifference import solve_waveequation1D
import utilities
import math
import time
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu') #OBS GPU DEACTIVATED

random.seed(8)

L = 1
Nx = 120
dt = 1e-3
N = 1000 #1000

sel = [0,-1]

Nc = 4000 #5000
Um = torch.zeros(Nc, 2, N+1, device=device)
Parm = torch.zeros(Nc, 3, device=device)

x = torch.linspace(0, L, Nx + 1, device=device)
dx = x[1] - x[0]
t = torch.linspace(0, N * dt, N + 1, device=device)
x = x.view(1,-1).repeat(N+1,1)
t = t.view(-1,1).repeat(1, Nx+1)

I = lambda x : x * 0
V = lambda x : x * 0
# f = lambda x, t : ((((2 - 4*x**3*math.pi**2 + 64*math.pi**2*x**2 + (-60*math.pi**2 + 2)*x)*t**2 
#                       - 2*x**2 + 2*x)*np.sin(2*math.pi*x) 
#                     + 8*np.cos(5*math.pi*x)*t**2*(x - 1/2)*math.pi*(x + 1))*np.cos(20*math.pi*t) 
#                     + 32*np.sin(8*math.pi*x)*t*math.pi*np.sin(11*math.pi*t)*x*(x - 1))*1e2




# f_exact = lambda x, t : (( 8*torch.cos(5*math.pi*x)*math.pi)*torch.cos(20*math.pi*t) 
#                     + 32*torch.sin(8*math.pi*x)*torch.sin(11*math.pi*t)*x*(x - 1))*1e2

# source term
f0 = 2e1
t0 = 0.1
# # Alternative 
# f0 = 1e1
# t0 = 0
def dirac(x):
    x = x*0
    x[:,(30,60,90)] = 1
    return x
f_exact = lambda x, t : 1e2 * dirac(x)*(f0**2 * (t-t0) * torch.exp(-f0**2*(t-t0)**2))

f=f_exact(x,t)
# f0 = 100

# def dirac(x):
#     x = x*0
#     x[0] = 1
#     return x

# f = lambda x, t : 1e5 * (f0**2 * t * torch.exp(-f0**2*t**2)) * dirac(x)



c, par = utilities.generate_c(Nx+1, 1, Nc)
x = torch.linspace(0, L, Nx+1, device=device)



start = time.perf_counter()
for i in range(Nc):
    U, t, cpu, T = solve_waveequation1D(I, V, f, c[i].view(-1), x, Nx, dt, N, bc='Neumann')
    
    Um[i,0,:] = U[:, sel[0]].cpu().detach()
    Um[i,1,:] = U[:, sel[-1]].cpu().detach()
    Parm[i, :] = par[i].cpu().detach()
    del U
    if i % 100 == 0:
        print(i)
    
end = time.perf_counter()
Um.requires_grad=True
Parm.requires_grad=True
print('Elapsed time to create dataset: ' + str(end-start))

torch.save(Um, 'measurement.pt')
torch.save(Parm, 'Parameters.pt')


# test source
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for i in range(Nc//2):
    ax.plot(t.cpu().detach().numpy(), Um[i,0].cpu().detach().numpy())
ax.set_title('$u$ at $x=0.1$ for different $v_{p}$')
ax.set_xlabel('$t$')
ax.set_ylabel('$u$')
fig.tight_layout()
plt.show()
plt.savefig('disp_differentc.svg')
    
#plt.plot(t.detach().numpy(), Um[1,0].detach().numpy())
#plt.plot(t.detach().numpy(), Um[2,0].detach().numpy())









