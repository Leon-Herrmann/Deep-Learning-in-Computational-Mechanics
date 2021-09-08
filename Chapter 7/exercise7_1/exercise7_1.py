from FiniteDifference import solve_waveequation1D
import torch
import utilities
from IterativeSolver import IterativeInverseSolverWaveEquation1D

device = torch.device('cpu') 
torch.manual_seed(20)

# Model parameters
L = 1 # Length of bar
Nx = 120 # Number of points in spatial dimension
dt = 5e-3 # Time step size
N = 300 # Number of time steps

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

# Neural network parameters
lr = 5e-3
epochs = 1000
c_weight = 1e0

# Generate grid
x, t = utilities.generateGrid2d(L, Nx, N, dt, device)

# Sample source term
f = f_exact(x, t)

# Sample wave velocity
c_m = x[0]*0 + 1.
#c_m[30:40] = c_m[30:40]*0.4 #non-differentiable
    
#for i in range(len(x[0])):
#    c_m[i] = Utilities.bump(x[0][i],1,0.1,0.2) #differentiable

c_m = torch.from_numpy(utilities.bump(x[0],1,0.1,0.2))

# Generate measurement 
um, tt = solve_waveequation1D(I, V, f, c_m, x[0], Nx, dt, N, bc='Neumann')
uin = um[:,sel].view(1,-1,N+1) 

# Create and train model
Test = IterativeInverseSolverWaveEquation1D(uin,sel,N,Nx,x,I,V,f,dt)
history_cpred, history_cost = Test.train(epochs,lr)

# Prediction
c_pred = Test.predict_c(x[0])
u_pred, tt = solve_waveequation1D(I, V, f,
                                  c_pred, x[0],
                                  Nx, dt, N, 
                                  bc='Neumann')

# Plots
utilities.plot_history(history_cost, filename='history')

utilities.plot_field(x, t, u_pred.cpu().detach(), N+1, Nx+1, title='Prediction of $u$', filename='predictionu')
utilities.plot_field(x, t, um, N+1, Nx+1, title='Solution of $u$', filename='solutionu')
utilities.plot_field(x, t, abs(u_pred.cpu().detach()-um), N+1, Nx+1, title='Absolute Error of $u$', filename='erroru')

utilities.plot_wavevelocity(x[0], c_pred, c_m, filename="IterativeFD")
     
# Animation of wave velocity during learning, check .gif file 
utilities.animate_history(x[0], history_cpred, c_m, epochs, filename="training_history_c") 