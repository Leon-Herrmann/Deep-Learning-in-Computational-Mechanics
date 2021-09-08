import torch
import time
import copy
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    """Initialize weights of neural network with xavier initialization."""
    if type(m) == torch.nn.Linear: #or type(m) == torch.nn.Conv1d: CHECK THIS
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)

class _Transition(torch.nn.Module):
    """Transistions from convolutional NN outputs to fully-connected NN outpus."""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.squeeze()

class DataDrivenSolver1D:
    """A class used for solving the inverse wave equation with a data-driven approach."""
    def __init__(self):
        """Construct a DataDrivenSolver for the 1D wave equation."""
        
        # Neural network 
        self.model = self.buildModel()
        
        # Training parameters assigned during training
        self.cost = None
        self.optimizer = None
        self.U_train = None
        self.P_train = None
        self.nc = None
        self.nc_ = None
        
    def buildModel(self):
        """Build a neural network."""
        model = torch.nn.Sequential()
        
        # Convolutional neural network
        model.add_module('conv1d0', torch.nn.Conv1d(2, 40, kernel_size=3, stride=2, padding=0))
        model.add_module('relu0', torch.nn.ReLU(inplace=True))
        model.add_module('conv1d1', torch.nn.Conv1d(40, 1, kernel_size=3, stride=2, padding=0))
        model.add_module('relu1', torch.nn.ReLU(inplace=True))
        
        # Transition
        model.add_module('transition', _Transition())
        
        # Fully-connected neural network
        model.add_module('linear0', torch.nn.Linear(249,1000)) 
        model.add_module('relu2',torch.nn.ReLU(inplace=True))
        model.add_module('linear1', torch.nn.Linear(1000,3))
        model.add_module('relu3',torch.nn.ReLU(inplace=True)) 
           
        # Initialize weights
        model.apply(init_weights)
        model.to(device)
        
        return model
        
    def predict(self, U):
        """Predict parameters of the wave velocity from a measurement."""
        return self.model(U)

    def cost_function(self, P_pred, P, nc):
        """Compute cost function."""
        temp = P_pred - P
        # Weight the last parameter c
        temp[:, -1] = temp[:, -1] * 1e1
        cost = torch.sum(temp**2) / nc
        return cost
    
    def closure(self):
        """Closure function for the optimizer."""
        self.optimizer.zero_grad()
        P_pred = self.model(self.U_train)
        self.cost = self.cost_function(P_pred, self.P_train, self.nc)
        self.cost.backward(retain_graph=True)
        return self.cost
        
    def train(self, epochs, lr, U_train, P_train, U_test, P_test, optimizer='Adam', **kwargs):
        """Train the model."""
        
        # Assign training data
        self.U_train = U_train
        self.P_train = P_train
        self.nc = len(U_train)
        self.nc_ = len(U_test)
        
        # Initialize history arrays
        history_cost = np.zeros(epochs)
        history_cost_ = np.zeros(epochs)
        
        # Set optimizer
        if optimizer == 'LBFGS':
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr, **kwargs)
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr, **kwargs)
            
        # Save the neural network parameters for the best cost function
        cost_best = 1e10
        parameters_best = copy.deepcopy(self.model.state_dict())
            
        start = time.perf_counter()
        start0 = start
        # Training loop
        for epoch in range(epochs):
            
            # Update neural network parameters
            self.optimizer.step(self.closure)
            
            # Save cost function for training and testing set
            history_cost[epoch] = self.cost.cpu().detach().numpy()
            P_test_pred = self.model(U_test)
            history_cost_[epoch] = self.cost_function(P_test_pred, P_test, self.nc_).cpu().detach().numpy()
            
            # Print training state
            if epoch % 100 == 0:
                end = time.perf_counter()
                string = "Epoch: {}/{}\t\tLoss = {:.3E}\n\t\t\tElapsed time:  {:2f}"
                print(string.format(epoch,epochs-1,self.cost.cpu().detach().numpy(),end-start))
                start = time.perf_counter()
                
            # Save the neural network parameters for the best cost function
            if self.cost < cost_best:
                cost_best = self.cost
                parameters_best = copy.deepcopy(self.model.state_dict())
                
        # Load the neural network parameters for the best cost function
        self.model.load_state_dict(parameters_best)
        
        end0 = time.perf_counter()
        training_time = end0-start0
        print("total training time: {:2f}".format(training_time))
        
        return history_cost, history_cost_  