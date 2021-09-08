"""
Finite Difference for the 1D wave equation
Reference: http://hplgit.github.io/INF5620/doc/notes/wave-sphinx/main_wave.html
"""

import time
import torch

device = torch.device('cpu')

def solve_waveequation1D(I, V, f, c, x, Nx, dt, N, bc='Neumann'):
    start = time.perf_counter()
    
    dx = x[1] - x[0]
    T = dt*N
    C = c * dt / dx
    if max(C)>1:
        print('Warning: Courant number is larger than 1')
    C2 = C ** 2
    t = torch.linspace(0, N * dt, N + 1, device=device)
    
    # Solution arrays
    U = torch.zeros([N + 1, Nx + 1], device=device)
    u = torch.zeros(Nx + 1, device=device)
    u_1 = torch.zeros(Nx + 1, device=device)
    u_2 = torch.zeros(Nx + 1, device=device)
    
    # Load initial conditions into u_1
    u_1[:] = I(x[:])
    U[0, :] = u_1[:]
        
    # Special formula for first time step
    n = 0
    
    if bc=='Dirichlet':
        # DIRICHLET BC
        # Update all inner points
        u[1:-1] = u_1[1:-1] + dt*V(x[1:-1]) + \
                  0.5*C2[1:-1]*(u_1[:-2] - 2*u_1[1:-1] + u_1[2:]) + \
                  0.5*dt**2*f[n, 1:-1]
        # Insert boundary conditions
        u[0] = 0
        u[Nx] = 0
    elif bc=='Neumann':
        # NEUMANN BC
        # Update all inner points
        u[1:-1] = u_1[1:-1] + dt*V(x[1:-1]) + \
                  0.5*C2[1:-1]*(u_1[:-2] - 2*u_1[1:-1] + u_1[2:]) + \
                  0.5*dt**2*f[n, 1:-1]
        # Insert boundary conditions
        u[0] = u_1[0] + dt*V(x[0]) + \
                0.5*C2[0]*(u_1[1] - 2*u_1[0] + u_1[1]) + \
                0.5*dt**2*f[n, 0]
        
        u[-1] = u_1[-1] + dt*V(x[-1]) + \
                0.5*C2[-1]*(u_1[-2] - 2*u_1[-1] + u_1[-2]) + \
                0.5*dt**2*f[n, -1]  
        
    
    U[1, :] = u[:] 
    
    u_2[:], u_1[:] = u_1, u
    
    for n in range(1, N):
        if bc=='Dirichlet':
            # DIRICHLET BC
            # Update all inner points at time t[n+1]       
            u[1:-1] = - u_2[1:-1] + 2*u_1[1:-1] + \
                            C2[1:-1]*(u_1[:-2] - 2*u_1[1:-1] + u_1[2:]) + \
                            dt**2*f[n, 1:-1]
                         
            # Insert boundary conditions
            u[0] = 0
            u[Nx] = 0
        
        elif bc=='Neumann':
            # NEUMANN BC
            # Update all inner points at time t[n+1]       
            u[1:-1] = - u_2[1:-1] + 2*u_1[1:-1] + \
                            C2[1:-1]*(u_1[:-2] - 2*u_1[1:-1] + u_1[2:]) + \
                            dt**2*f[n, 1:-1]
                         
            # Insert boundary conditions
            # u[0] = u_1[0] + C2[0]*(u_1[1] - 2*u_1[0] + u_1[1])
                   
            # u[Nx] = u_1[-1] + C2[-1]*(u_1[-2] - 2*u_1[-1] + u_1[-2])
            
            u[0] = - u_2[0] + 2*u_1[0] + \
                   C2[0]*(u_1[1] - 2*u_1[0] + u_1[1]) + \
                   dt**2*f[n, 0]

            u[-1] = - u_2[-1] + 2*u_1[-1] + \
                   C2[-1]*(u_1[-2] - 2*u_1[-1] + u_1[-2]) + \
                   dt**2*f[n, -1]                   
            
        # Switch variables before next step
        u_2[:], u_1[:] = u_1, u
        U[n + 1, :] = u[:]
        
        cpu = time.perf_counter() - start
    return U, t, cpu, T

