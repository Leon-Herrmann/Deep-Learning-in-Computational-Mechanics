import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(2)

#Load data
E_data = torch.from_numpy(np.genfromtxt('setup_minibatch/E.txt', delimiter=', ')).float().to(device)
eps11_data = torch.from_numpy(np.genfromtxt('setup_minibatch/eps11.txt', delimiter=', ')).float().to(device)
eps22_data = torch.from_numpy(np.genfromtxt('setup_minibatch/eps22.txt', delimiter=', ')).float().to(device)
eps12_data = torch.from_numpy(np.genfromtxt('setup_minibatch/eps12.txt', delimiter=', ')).float().to(device)
#f = torch.from_numpy(np.genfromtxt('setup/f.txt', delimiter=', '))

# scale E_data between 0 and 1
E_scale = torch.max(E_data)
E_data = E_data / E_scale

n = int(np.sqrt(E_data.shape[1]))

E_data = E_data.reshape(-1,1,n,n) 
eps11_data = eps11_data.reshape(-1,1,n,n)
eps22_data = eps22_data.reshape(-1,1,n,n)
eps12_data = eps12_data.reshape(-1,1,n,n)
eps_data = torch.cat((eps11_data, eps22_data, eps12_data),1)

samples = E_data.shape[0]

samples_train = int(np.ceil(samples*0.9))
mini_size = 50
mini_batches = int(np.ceil(samples_train / mini_size))
samples_val = samples - samples_train
E_data_train = E_data[:samples_train]
E_data_val = E_data[samples_train:]
eps_data_train = eps_data[:samples_train]
eps_data_val = eps_data[samples_train:]
E_data_train.requires_grad = True

class ConvAutoencoder(torch.nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__() # derives from torch.nn.Module
       
        # Encoder
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv1 = torch.nn.Conv2d(1, 32, 3, stride=1, padding=1, padding_mode='zeros')
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1, padding_mode='zeros')
        
        # FC        
        self.linear1 = torch.nn.Linear(8**2*64,1024)
        self.linear2 = torch.nn.Linear(1024, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 256)
        self.linear5 = torch.nn.Linear(256, 1024) 
        self.linear6 = torch.nn.Linear(1024, 8**2*64)
    
        # Decoder
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.t_conv1 = torch.nn.Conv2d(128, 32, 3, stride=1, padding=1, bias=True)
        self.t_conv2 = torch.nn.Conv2d(65, 32, 3, stride=1, padding=1, bias=True)
        
        self.t_conv3 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=1, bias=True)
        self.t_conv4 = torch.nn.Conv2d(16, 3, 3, stride=1, padding=1, bias=True)
        
        # Batchnorm
        self.bnorm16 = torch.nn.BatchNorm2d(16)
        self.bnorm32 = torch.nn.BatchNorm2d(32)
        self.bnorm64 = torch.nn.BatchNorm2d(64)
        
#        self.dropout = torch.nn.Dropout(0.1, inplace=True)
    
    def forward(self, x):
        # Encoder
        y0 = x
        x = self.conv1(x)
        x = self.pool(x)
        x = self.bnorm32(x)
        x = torch.nn.LeakyReLU(inplace=True)(x)
        y1 = x
        x = self.conv2(x)
        x = self.pool(x)
        x = self.bnorm64(x)
        x = torch.nn.LeakyReLU(inplace=True)(x)
        y2 = x
        
        # FC
        x = x.reshape(-1, 8**2*64)
        x = torch.nn.LeakyReLU(inplace=True)(self.linear1(x))
        x = torch.nn.LeakyReLU(inplace=True)(self.linear2(x))
        x = torch.nn.LeakyReLU(inplace=True)(self.linear3(x))
        x = torch.nn.LeakyReLU(inplace=True)(self.linear4(x))
        x = torch.nn.LeakyReLU(inplace=True)(self.linear5(x))
        x = torch.nn.LeakyReLU(inplace=True)(self.linear6(x))
        x = x.reshape(-1, 64, 8, 8)
            
        # Decoder  
        x = torch.cat((x, y2), axis=1)
        x = self.upsample(x)
        x = self.t_conv1(x)
        x = self.bnorm32(x)
        x = torch.nn.LeakyReLU(inplace=True)(x)
        x = torch.cat((x, y1), axis=1)
        x = self.upsample(x)
        x = torch.cat((x, y0), axis=1)
        x = self.t_conv2(x)
        
        x = self.bnorm32(x)
        x = torch.nn.LeakyReLU(inplace=True)(x)
        x = self.t_conv3(x)
        
        x = self.bnorm16(x)
        x = torch.nn.LeakyReLU(inplace=True)(x)
        x = self.t_conv4(x)
     
#        x = self.dropout(x)
        
        return x
        
model = ConvAutoencoder()
model.to(device) #???? check which ones


print(model.forward(E_data_train).shape)

# training
epochs = 200
lr = 1e-3
weight_decay = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
cost_arr_train = np.zeros(epochs)
cost_arr_val = np.zeros(epochs)
best_cost = 1e10

for epoch in range(epochs):
    for mini_batch in range(mini_batches):
        eps_pred_train = model.forward(E_data_train[mini_batch*mini_size:(mini_batch+1)*mini_size])
        cost_train = torch.sum((eps_pred_train - eps_data_train[mini_batch*mini_size:(mini_batch+1)*mini_size])**2)/(n**2*samples_train) 
        cost_arr_train[epoch] += cost_train.cpu().detach().numpy()
        optimizer.zero_grad()
        cost_train.backward()
        optimizer.step()
    eps_pred_val = model.forward(E_data_val)
    cost_val = torch.sum((eps_pred_val - eps_data_val)**2)/(n**2*samples_val)
    cost_arr_val[epoch] = cost_val.cpu().detach().numpy()
    if cost_val < best_cost:
        model_parameters_best = copy.deepcopy(model.state_dict())
        best_cost = cost_arr_val[epoch]
    if epoch % 10 == 0: # early stopping could be added
        print(f'Epoch: {epoch}, Training Cost: {cost_arr_train[epoch]}')
        print(f'\t Validation Cost: {cost_arr_val[epoch]}')

model.load_state_dict(model_parameters_best)

# postprocessing
        
x = np.linspace(0,1,n)
y = np.linspace(0,1,n)
x, y = np.meshgrid(x, y)
levels = np.linspace(0, 0.4, 12) 

i = 0
j = 0
fig, ax = plt.subplots()
ax.set_aspect('equal')
cp = ax.pcolormesh(model.forward(E_data_train)[i,j].cpu().detach(), cmap=plt.cm.jet) #levels=levels
fig.colorbar(cp)
plt.show()

fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')
cp = ax.pcolormesh(eps_data_train[i,j].cpu().detach(), cmap=plt.cm.jet)
fig.colorbar(cp)
plt.show()

#
###fig, ax = plt.subplots()
###ax.set_aspect('equal')
###cp = ax.contourf(((model.forward(E_data)[i,j] - eps11_data[i,j])).cpu().detach(), cmap=plt.cm.jet) #levels=levels
###fig.colorbar(cp)
###plt.show()

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.plot(cost_arr_train, 'k',label="Training Cost")
ax.plot(cost_arr_val, 'r',label="Validation Cost")
ax.legend()
plt.show()


##stress computation
#E = E_data_train[i].unsqueeze(0).cpu().detach()
#E.requires_grad = False
#eps = model.forward(E)
#E = E * E_scale
#nu = E
#nu[nu>85000-1e-1] = 0.22
#nu[nu<3000+1e-1] = 0.4
#
#C = torch.zeros(3, 3, 32, 32)
#C[0, 0] += 1.
#C[1, 0] = nu
#C[0, 1] = nu
#C[1, 1] += 1.
#C[2, 2] = (1-nu)/2.
#C *= E/(1-nu**2)
#
#sig = torch.matmul(torch.permute(C,(2,3,0,1)), torch.permute(eps, (2,3,1,0)))
#sig = torch.permute(sig, (3,2,0,1))
#
#fig, ax = plt.subplots()
#ax.set_aspect('equal', 'box')
#cp = ax.pcolormesh(sig[0,j].cpu().detach(), cmap=plt.cm.jet)
#fig.colorbar(cp)
#plt.show()
