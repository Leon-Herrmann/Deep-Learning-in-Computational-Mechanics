import torch
import time
import copy
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    if type(m) == torch.nn.Linear: # or type(m) == torch.nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        #torch.nn.init.kaiming_uniform_(m.weight)
        #torch.nn.init.xavier_uniform(m.bias)
        #torch.nn.init.xavier_uniform_(m.bias, gain=torch.nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)

class _Transition(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.squeeze()

class DataDrivenSolver1D:
    def __init__(self):
    
        self.model = torch.nn.Sequential()
        
        #self.model.add_module('bn-1', torch.nn.BatchNorm1d(2))
        
        self.model.add_module('conv1d0', torch.nn.Conv1d(2, 40, kernel_size=3, stride=2, padding=0))
        #self.model.add_module('bn0', torch.nn.BatchNorm1d(40)) 
        self.model.add_module('relu0', torch.nn.ReLU(inplace=True))
        self.model.add_module('conv1d1', torch.nn.Conv1d(40, 1, kernel_size=3, stride=2, padding=0))
        #self.model.add_module('bn1', torch.nn.BatchNorm1d(1)) 
        self.model.add_module('relu1', torch.nn.ReLU(inplace=True))
        
        self.model.add_module('transition', _Transition())
        self.model.add_module('linear0', torch.nn.Linear(249,2000)) #249
        self.model.add_module('Tanh0', torch.nn.Tanh())
        #self.model.add_module('relu2',torch.nn.ReLU(inplace=True))
        self.model.add_module('linear1', torch.nn.Linear(2000,3))
        
        self.model.apply(init_weights)
        self.model.to(device)
        
        self.cost = None
        self.optimizer = None
        self.U_train = None
        self.P_train = None
        self.U_test = None
        self.P_test = None
        self.nc = None
        self.nc_ = None
        
    def predict(self, U):
        return self.model(U)

    def cost_function(self, P_pred, P, nc):
        temp = P_pred - P
        temp[:, -1] = temp[:, -1] * 1e1
        cost = torch.sum(temp**2) / nc
        return cost
    
    def closure(self):
        self.optimizer.zero_grad()
        P_pred = self.model(self.U_train)
        self.cost = self.cost_function(P_pred, self.P_train, self.nc)
        self.cost.backward(retain_graph=True)
        return self.cost
        
    def train(self, epochs, lr, patience, lr_factor, U_train, P_train, U_test, P_test, optimizer='Adam'):
        self.U_train = U_train
        self.P_train = P_train
        self.nc = len(U_train)
        self.nc_ = len(U_test)
        history_cost = np.zeros(epochs)
        history_cost_ = np.zeros(epochs)
        
        if optimizer == 'LBFGS':
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr)
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
            
        cost_best = 1e10
        parameters_best = copy.deepcopy(self.model.state_dict())
            
        start = time.perf_counter()
        start0 = start
        for epoch in range(epochs):
            self.optimizer.step(self.closure)
            history_cost[epoch] = self.cost.cpu().detach().numpy()
            
            
            P_test_pred = self.model(U_test)
            history_cost_[epoch] = self.cost_function(P_test_pred, P_test, self.nc_).cpu().detach().numpy()
            
            if epoch % 100 == 0:
                end = time.perf_counter()
                string = "Epoch: {}/{}\t\tLoss = {:.3E}\n\t\t\tElapsed time:  {:2f}"
                print(string.format(epoch,epochs-1,self.cost.cpu().detach().numpy(),end-start))
                start = time.perf_counter()
                
            if self.cost < cost_best:
                cost_best = self.cost
                parameters_best = copy.deepcopy(self.model.state_dict())
                
            # if epoch==800:
            #     lr = lr*0.5
            #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
                    
        self.model.load_state_dict(parameters_best)
        end0 = time.perf_counter()
        training_time = end0-start0
        print("total training time: {:2f}".format(training_time))
        return history_cost, history_cost_
        
        
                