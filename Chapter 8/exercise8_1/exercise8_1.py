import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(2)

#Load data
E_data = torch.from_numpy(np.genfromtxt('E.txt', delimiter=', ')).float().to(device)
eps11_data = torch.from_numpy(np.genfromtxt('eps11.txt', delimiter=', ')).float().to(device)
eps22_data = torch.from_numpy(np.genfromtxt('eps22.txt', delimiter=', ')).float().to(device)
eps12_data = torch.from_numpy(np.genfromtxt('eps12.txt', delimiter=', ')).float().to(device)
#f = torch.from_numpy(np.genfromtxt('f.txt', delimiter=', '))

n = int(np.sqrt(E_data.shape[1]))
samples = E_data.shape[0]

#E_data = E_data[0]
#eps11_data = eps11_data[0]
#eps22_data = eps22_data[0]
#eps12_data = eps12_data[0]
#samples = 1

E_data = E_data.reshape(samples,1,n,n) #OBS GENERALIZE
eps11_data = eps11_data.reshape(samples,1,n,n)
eps22_data = eps22_data.reshape(samples,1,n,n)
eps12_data = eps12_data.reshape(samples,1,n,n)
eps_data = torch.cat((eps11_data, eps22_data, eps12_data),1)

samples_train = int(np.ceil(samples/2))
samples_val = samples - samples_train
E_data_train = E_data[:samples_train]
E_data_val = E_data[samples_train:]
eps_data_train = eps_data[:samples_train]
eps_data_val = eps_data[samples_train:]
E_data_train.requires_grad = True

class ConvAutoencoder(torch.nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=2, padding=0)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=0)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=0)

        #FC        
        self.linear1 = torch.nn.Linear(144,500)
        self.linear2 = torch.nn.Linear(500,500)
        self.linear3 = torch.nn.Linear(500, 6400)
    
        #Decoder
        self.upsample1 = torch.nn.Upsample(scale_factor=1.4, mode='bilinear')
        self.t_conv1 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=0, bias=True)
        self.upsample2 = torch.nn.Upsample(scale_factor=1.3, mode='bilinear')
        self.t_conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=0, bias=True)
        self.upsample3 = torch.nn.Upsample(scale_factor=1.3, mode='bilinear')
        self.t_conv3 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=0, bias=True)
        self.upsample4 = torch.nn.Upsample(size=52, mode='bilinear')
        self.t_conv4 = torch.nn.Conv2d(16, 3, 3, stride=1, padding=0, bias=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.BatchNorm2d(16)(x)
        x = torch.nn.LeakyReLU()(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.nn.BatchNorm2d(32)(x)
        x = torch.nn.LeakyReLU()(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = torch.nn.BatchNorm2d(16)(x)
        x = torch.nn.LeakyReLU()(x)
 
        x = x.reshape(-1,144)
        x = self.linear1(x)
        x = torch.nn.LeakyReLU()(x)
#        x = self.linear2(x)
#        x = torch.nn.LeakyReLU()(x)
        x = self.linear3(x)
        x = torch.nn.LeakyReLU()(x)
        x = x.reshape(-1,16,20,20)
        #16x14x14
        
        x = self.upsample1(x)
        x = self.t_conv1(x)
        x = torch.nn.BatchNorm2d(32)(x)
        x = torch.nn.LeakyReLU()(x)
        x = self.upsample2(x)
        x = self.t_conv2(x)
        x = torch.nn.BatchNorm2d(32)(x)
        x = torch.nn.LeakyReLU()(x)
        x = self.upsample3(x)
        x = self.t_conv3(x)
        x = torch.nn.BatchNorm2d(16)(x)
        x = torch.nn.LeakyReLU()(x)
        x = self.upsample4(x)
        x = self.t_conv4(x)
        
        return x
        
model = ConvAutoencoder()
model.to(device)

print(model.forward(E_data_train).shape)

# training
epochs = 5000
lr = 2e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
cost_arr_train = torch.zeros(epochs)
cost_arr_val = torch.zeros(epochs)

for epoch in range(epochs):
    eps_pred_train = model.forward(E_data_train)
    eps_pred_val = model.forward(E_data_val)
    cost_train = torch.sum((eps_pred_train - eps_data_train)**2)/(n**2*samples_train) 
    cost_val = torch.sum((eps_pred_val - eps_data_val)**2)/(n**2*samples_val)
    cost_arr_train[epoch] = cost_train
    cost_arr_val[epoch] = cost_val
    optimizer.zero_grad()
    cost_train.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Training Cost: {cost_train.detach().numpy()}')
        print(f'\t Validation Cost: {cost_val.detach().numpy()}')
        
x = np.linspace(0,1,n)
y = np.linspace(0,1,n)
x, y = np.meshgrid(x, y)
levels = np.linspace(0, 0.4, 12) 

i = 0
j = 0
fig, ax = plt.subplots()
ax.set_aspect('equal')
cp = ax.contourf(model.forward(E_data_train)[i,j].detach(), levels=levels, cmap=plt.cm.jet) #levels=levels
fig.colorbar(cp)
plt.show()

fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')
cp = ax.contourf(eps_data_train[i,j].detach(), levels=levels, cmap=plt.cm.jet)
fig.colorbar(cp)
plt.show()
#
###fig, ax = plt.subplots()
###ax.set_aspect('equal')
###cp = ax.contourf(((model.forward(E_data)[i,j] - eps11_data[i,j])).detach(), cmap=plt.cm.jet) #levels=levels
###fig.colorbar(cp)
###plt.show()
#
#fig, ax = plt.subplots()
#ax.set_yscale('log')
#ax.plot(cost_arr_train.detach(), 'k',label="Training Cost")
#ax.plot(cost_arr_val.detach(), 'r',label="Validation Cost")
#ax.legend()
#plt.show()
