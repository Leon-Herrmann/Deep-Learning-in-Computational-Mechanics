import torch
import matplotlib.pyplot as plt
from DataDrivenSolver import DataDrivenSolver1D
import utilities

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(2)

# Load data
U_data = torch.load('Measurement.pt')
P_data = torch.load('Parameters.pt')
U_data = U_data.to(device)
P_data = P_data.to(device)
nc_tot = len(U_data)

# Split data into training and testing set
nc = nc_tot // 8
U_test = U_data[:nc]
P_test = P_data[:nc]
U_train = U_data[nc:]
P_train = P_data[nc:]
            
# Neural network parameters
epochs = 2000
lr = 1e-3
            
# Create and train model
test = DataDrivenSolver1D()
history_cost, history_cost_ = test.train(epochs, lr, U_train, P_train, U_test=U_test, P_test=P_test, optimizer='Adam')

# Compute cost function of entire training and testing set
train_cost = test.cost_function(test.predict(U_train), P_train, len(U_train)).cpu().detach().numpy()
test_cost = test.cost_function(test.predict(U_test), P_test, len(U_test)).cpu().detach().numpy()
print('Training cost: {:2f} \tTesting cost: {:2f}'.format(train_cost,test_cost))

# Predictions from the test set
P_test_pred = test.model(U_test)
P_train_pred = test.model(U_train)

# Plots
utilities.plot_wavevelocities(P_test_pred, P_test, 3, shift=40, title='Test set', filename='test_set')
utilities.plot_wavevelocities(P_train_pred, P_train, 3, 10, title='Training set', filename='train_set')

utilities.plot_history(history_cost, history_cost_, filename='history_cost')