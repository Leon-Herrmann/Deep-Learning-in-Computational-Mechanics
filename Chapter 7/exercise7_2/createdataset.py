import torch
from FiniteDifference import solve_waveequation1D
import utilities
import time
import random

device = torch.device('cpu')

random.seed(8)

# Model parameters
L = 1 # Length of bar
Nx = 120 # Number of points in spatial dimension
dt = 1e-3 # Time step size
N = 1000 # Number of time steps

# Position of sensors
sel = (0,-1)

# Number of samples
Nc = 4000

# Arrays to save the measurement data and the corresponding wave velocity
Um = torch.zeros(Nc, 2, N+1, device=device)
Parm = torch.zeros(Nc, 3, device=device)

# Generate grid
x, t = utilities.generateGrid2d(L, Nx, N, dt, device)

# Source term
f0 = 2e1
t0 = 0.1
def dirac(x):
    x = x*0
    x[:,(30,60,90)] = 1
    return x
f_exact = lambda x, t : 1e2 * dirac(x)*(f0**2 * (t-t0) * torch.exp(-f0**2*(t-t0)**2))

# Sample source term
f = f_exact(x,t)

I = lambda x : x * 0
V = lambda x : x * 0

# Generate wave velocities
c, par = utilities.generate_c(Nx+1, 1, Nc, x[0])

x = torch.linspace(0, L, Nx+1, device=device)

start = time.perf_counter()
# Loop over the wave velocities
for i in range(Nc):
    # Solve the wave equation 
    U, t = solve_waveequation1D(I, V, f, c[i].view(-1), x, Nx, dt, N, bc='Neumann')
    
    # Save the solution at the sensor positions
    Um[i,0,:] = U[:, sel[0]].cpu().detach()
    Um[i,1,:] = U[:, sel[-1]].cpu().detach()
    Parm[i, :] = par[i].cpu().detach()
    del U
    
    if i % 100 == 0:
        print(i)
    
end = time.perf_counter()
print('Elapsed time to create dataset: ' + str(end-start))

torch.save(Um, 'Measurement.pt')
torch.save(Parm, 'Parameters.pt')