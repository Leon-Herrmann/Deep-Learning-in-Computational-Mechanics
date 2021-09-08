from FiniteDifference import solve_waveequation1D
import torch
import time
import numpy as np

def init_weights(m):
    """Initialize weights of neural network with xavier initialization."""
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)

class IterativeInverseSolverWaveEquation1D:
    """A class used for solving the inverse wave equation iteratively."""
    
    def __init__(self,uin,sel,N,Nx,x,I,V,f,dt):
        """Construct a IterativeInverseSolver for the 1D wave equation."""
        
        # Model parameters 
        self.uin = uin # Measurement of u
        self.sel = sel # Index of measurement location on spatial domain
        self.N = N # Number of timesteps
        self.Nx = Nx # Number of spatial points
        self.x = x # Spatial grid
        self.I = I # Initial conditions 1
        self.V = V # Initial conditions 2
        self.f = f # Source term
        self.dt = dt # Step size
        self.model = self.buildModel(1,[100,100],1)
        
        # Training parameters assigned during training
        self.c_weight = None
        self.cost = None
        self.c_pred = None
        self.optimizer = None
        
    def buildModel(self, input_dimension, hidden_dimension, output_dimension):
        """Build a neural network of given dimensions."""
    
        modules = []
        modules.append(torch.nn.Linear(input_dimension, hidden_dimension[0]))
        modules.append(torch.nn.LeakyReLU())
        for i in range(len(hidden_dimension) - 1):
            modules.append(torch.nn.Linear(hidden_dimension[i], hidden_dimension[i + 1]))
            modules.append(torch.nn.LeakyReLU())
    
        modules.append(torch.nn.Linear(hidden_dimension[-1], output_dimension))
        # Scale ouptut between 0 and 1 with Sigmoid
        modules.append(torch.nn.Sigmoid())
        
        model = torch.nn.Sequential(*modules)
        
        # Initialize weights
        model.apply(init_weights)
    
        return model
      
    def predict_c(self,x):
        """Predict wave velocity from a spatial coordinate."""
        c = self.model(x.view(-1,1)).view(-1)
        return c
        
    def costFunction(self, u_pred, c_pred):
        """Compute cost function."""
        cost = (u_pred[:,self.sel] - self.uin.view(self.N+1,-1))**2
        cost = torch.sum(cost,axis=0)/len(self.sel)/(self.N+1) 
        cost = torch.sum(cost)
        cost += 0.5*((c_pred[0]-1.)**2 + (c_pred[-1]-1.)**2)*self.c_weight
        return cost
    
    def closure(self):
        """Closure function for the optimizer."""
        self.optimizer.zero_grad()
        self.c_pred = self.predict_c(self.x[0])
        u_pred, tt = solve_waveequation1D(self.I, self.V, self.f,
                                                  self.c_pred, self.x[0],
                                                  self.Nx, self.dt, self.N, 
                                                  bc='Neumann')
        self.cost = self.costFunction(u_pred, self.c_pred)
        self.cost.backward(retain_graph=True)
        return self.cost
    
    def train(self, epochs, lr, c_weight=1, **kwargs):
        """Train the model."""
        
        # Set training parameters
        self.c_weight = c_weight
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, **kwargs)
        
        # Initialize history arrays
        history_cost = np.zeros(epochs)
        history_cpred = np.zeros(((epochs // 10)+1,self.Nx+1))
        
        # Track time
        start0 = time.perf_counter()
        start = start0
        
        # Training loop
        for epoch in range(epochs):
            
            # Learning rate scheduler
            if epoch % 300 == 0 and epoch!=0:
                lr = lr*0.1
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr, **kwargs)
                print("learning rate scheduler")
            
            # Update neural network parameters
            self.optimizer.step(self.closure)
            
            # Save c for every 10th epoch for the animation 
            # Print training state
            if epoch % 10 == 0:
                history_cpred[(epoch // 10)] = self.c_pred.cpu().detach().numpy()
                elapsed_time = time.perf_counter() - start
                string = "Epoch: {}/{}\t\tCost function = {:.3E}\t\tElapsed time:  {:2f}"
                print(string.format(epoch, epochs-1, self.cost.cpu().detach().numpy(), elapsed_time))
                start = time.perf_counter()

            # Save cost function history
            history_cost[epoch] = self.cost.cpu().detach().numpy()
            
        # Save c for last prediction
        history_cpred[-1] = self.c_pred.cpu().detach().numpy()
        
        # Print total training time
        print("Total training time: {:2f}".format(time.perf_counter()-start0))
        
        return history_cpred, history_cost