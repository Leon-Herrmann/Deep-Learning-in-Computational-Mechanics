from FiniteDifference import solve_waveequation1D
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import math
import copy
import utilities

class IterativeInverseSolverWaveEquation1D:
    """
    A class used for OBS
    """
    
    def __init__(self,uin,sel,N,Nx,x,I,V,f,dt): #TODO CHANGE x 
        """Construct a IterativeInverseSolver for the 1D wave equation."""
        
        self.uin = uin
        self.sel = sel
        self.N = N
        self.Nx = Nx
        self.x = x
        self.I = I
        self.V = V
        self.f = f
        self.dt = dt
        
        
        self.c_weight = None
        self.cost = None
        self.c_pred = None
        self.optimizer = None
        
    def init_weights(self,m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
            #torch.nn.init.kaiming_uniform_(m.weight)
            #torch.nn.init.xavier_uniform(m.bias)
            #torch.nn.init.xavier_uniform_(m.bias, gain=torch.nn.init.calculate_gain('relu'))
            m.bias.data.fill_(0.01)
        
    def buildModel(self): #TODO GENERALIZE
        """Build a neural network of given dimensions."""
        
        self.model = torch.nn.Sequential()
        
        self.model.add_module('linear0', torch.nn.Linear(1,100))
        self.model.add_module('leakyRelu0', torch.nn.LeakyReLU())
        self.model.add_module('linear1', torch.nn.Linear(100,100))
        self.model.add_module('leakyReLU1', torch.nn.LeakyReLU())
        self.model.add_module('linear2', torch.nn.Linear(100,1))
        self.model.add_module('Tanh2', torch.nn.Tanh())
        
        #torch.nn.init.xavier_uniform_(self.model[0].weight, gain=1.0)
        self.model.apply(self.init_weights)
        

    def predict_c(self,x):
        """Predict wave velocity."""
        c = (1.-abs(self.model(x.view(-1,1)))).view(-1)
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
        #TODO OBS CHANGE OUTPUTS!!!!!
        u_pred, tt, cpu, T = solve_waveequation1D(self.I, self.V, self.f,
                                                  self.c_pred, self.x[0],
                                                  self.Nx, self.dt, self.N, 
                                                  bc='Neumann')
        self.cost = self.costFunction(u_pred, self.c_pred)
        self.cost.backward(retain_graph=True)
        return self.cost
    
    def train(self,epochs,lr,c_weight=1):
        """Train the model."""
        
        self.c_weight = c_weight
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        #self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=1e-2)
        history_cost = np.zeros(epochs)
        history_cpred = np.zeros(((epochs // 10)+1,self.Nx+1))
        
        k = 0 #REMOVE
        
        start0 = time.perf_counter()
        start = start0
        for epoch in range(epochs):
            
            # learning rate scheduler
            if epoch % 500 == 0 and epoch!=0:
                lr = lr*0.1
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                print("learning rate scheduler")
            
            self.optimizer.step(self.closure)
            
            if epoch % 10 == 0:
                history_cpred[k] = self.c_pred.cpu().detach().numpy()
                k += 1 #TODO REMOVE k (calculate from epoch)
                elapsed_time = time.perf_counter() - start
                string = "Epoch: {}/{}\t\tCost function = {:.3E}\t\tElapsed time:  {:2f}"
                print(string.format(epoch, epochs-1, self.cost.cpu().detach().numpy(), elapsed_time))
                start = time.perf_counter()

            history_cost[epoch] = self.cost.cpu().detach().numpy()
            
        history_cpred[-1] = self.c_pred.cpu().detach().numpy()
        print("Total training time: {:2f}".format(time.perf_counter()-start0))
        
        return history_cpred, history_cost