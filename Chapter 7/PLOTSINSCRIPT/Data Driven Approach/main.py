import torch
import matplotlib.pyplot as plt
from DataDrivenSolver import DataDrivenSolver1D
import utilities

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(2)

U_data = torch.load('Measurement.pt')
P_data = torch.load('Parameters.pt')
U_data = U_data.to(device)
P_data = P_data.to(device)
nc_tot = len(U_data)
nc = nc_tot // 8

U_test = U_data[:nc]
P_test = P_data[:nc]
U_train = U_data[nc:]
P_train = P_data[nc:]
            
epochs = 2000 #2000
lr = 5e-3
patience = 300 #????
lr_factor = 0.5
            
test = DataDrivenSolver1D()
history_cost, history_cost_ = test.train(epochs, lr, patience, lr_factor, U_train, P_train, U_test=U_test, P_test=P_test, optimizer='Adam')

train_cost = test.cost_function(test.predict(U_train), P_train, len(U_train)).cpu().detach().numpy()
test_cost = test.cost_function(test.predict(U_test), P_test, len(U_test)).cpu().detach().numpy()

print(train_cost)
print(test_cost)

# example test cases
P_pred = test.model(U_test)
P_pred[:,0:2] = torch.round(P_pred[:,0:2]) #OBS ROUNDING NOT IN TRAINING
ab = P_pred[:,0:2].long().cpu().detach().numpy()
ab_test = P_test[:,0:2].long().cpu().detach().numpy()


A = 3
fig, ax = plt.subplots(A,A)
#fig.set_title('Examples from test set')
x = torch.linspace(0,1,120, device=device)

s = 10
for i in range(A):
    for j in range(A):    
        n = A*j+i+s
        c_pred = x*0 + 1
        c_pred[ab[n,0]:ab[n,1]] = c_pred[ab[n,0]:ab[n,1]] * 0 + P_pred[n,2]
        c_test = torch.ones(120, device=device)
        c_test[ab_test[n,0]:ab_test[n,1]] = c_pred[ab_test[n,0]:ab_test[n,1]] * 0 + P_test[n,2]
        
        #ax[i,j].grid(which='both')
        if i==0 and j==0:
            ax[i,j].plot(x.cpu().detach().numpy(),c_pred.cpu().detach().numpy(), color='k', label='Prediction')
            ax[i,j].plot(x.cpu().detach().numpy(),c_test.cpu().detach().numpy(), color='r', linestyle='--', label='Solution')
        else:
            ax[i,j].plot(x.cpu().detach().numpy(),c_pred.cpu().detach().numpy(), color='k')
            ax[i,j].plot(x.cpu().detach().numpy(),c_test.cpu().detach().numpy(), color='r', linestyle='--')
        ax[i,j].set_xlabel('$x$')
        ax[i,j].set_ylabel('$v_{p}$')
        
fig.tight_layout() 
fig.subplots_adjust(top=0.85)
fig.suptitle('Examples from the test set',y=0.98)
fig.legend(bbox_to_anchor=(1.015,1.02))
plt.show()
plt.savefig('test_set.png')
plt.savefig('test_set.eps')

utilities.plot_history(history_cost, history_cost_, filename='history_cost')

#test prediction time
import time
start = time.perf_counter()
test.model(U_test)
#test.model(U_test[0].view(1,2,-1))
end = time.perf_counter()
print("prediction time: "+str(end-start))




   