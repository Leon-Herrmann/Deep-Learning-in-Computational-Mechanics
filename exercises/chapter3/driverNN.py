import torch
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import NeuralNetwork
import datetime
import copy

selectNN = 2 # 0 for UNet, 1 for sequential CNN, 2 for UNet with subsequent feedforward CNN
seed = 2
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter(log_dir="./logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# hyperparameters
batchSize = 128  
alpha = -0.2
beta = 0.2
weightDecay = 0 #1e-3 #1e-3 
lr = 2e-3 
epochs = 4000

if weightDecay == 0:
    earlyStopping = False
else:
    earlyStopping = True
    
kernelSize = 3 

if selectNN == 0:
    channels = [1, 32, 64] 
    channelsOut = 3 
    numberOfConvolutionsPerBlock = 1 
    model = NeuralNetwork.UNet(channels, channelsOut, numberOfConvolutionsPerBlock, kernelSize)
elif selectNN == 1:
    channels = [1, 32, 64, 32]
    channelsOut = 3
    model = NeuralNetwork.FeedforwardCNN(channels, channelsOut, kernelSize)
elif selectNN == 2:
    channelsUNet = [1, 32, 64]
    numberOfConvolutionsPerBlockUNet = 1
    channelsFeedforwardCNN = [64, 32, 16]
    channelsOut = 3
    model = NeuralNetwork.UNetWithSubsequentFeedforwardCNN(channelsUNet, numberOfConvolutionsPerBlockUNet, channelsFeedforwardCNN, channelsOut, kernelSize)

numberOfTrainingSamples = 1 #128 
numberOfSamples = numberOfTrainingSamples + 32 # TODO set maximum
dataset = NeuralNetwork.elasticityDataset(device, numberOfSamples)
# normalization
dataset.E = (dataset.E - np.mean([3000, 85000]))/np.std([3000, 85000])
datasetTraining, datasetValidation = torch.utils.data.random_split(dataset, [numberOfTrainingSamples, len(dataset) - numberOfTrainingSamples], generator=torch.Generator().manual_seed(2))

dataloaderTraining = DataLoader(datasetTraining, batch_size=batchSize)
dataloaderValidation = DataLoader(datasetValidation, batch_size=len(dataset))

#model.apply(NeuralNetwork.initWeights) # TODO CHECK
model.to(device)
summary(model, (1, 32, 32))

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weightDecay)
lr_lambda = lambda epoch : (beta*epoch + 1)**alpha
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, verbose=False)
trainingCostHistory = np.zeros(epochs)
validationCostHistory = np.zeros(epochs)
start = time.perf_counter()
start0 = start
bestCost = 1e10

for epoch in range(epochs):

    model.train()
    for batch, sample in enumerate(dataloaderTraining):      
        optimizer.zero_grad()

        prediction = model(sample[0])
        cost = NeuralNetwork.costFunction(prediction, sample[1])
        
        trainingCostHistory[epoch] += cost.detach() * len(sample[1])
        
        cost.backward()
        optimizer.step()
                
    trainingCostHistory[epoch] /= numberOfTrainingSamples
    scheduler.step()
    
    model.eval()
    sample = next(iter(dataloaderValidation))
    with torch.no_grad():
        prediction = model(sample[0])
        cost = NeuralNetwork.costFunction(prediction, sample[1])
        
        validationCostHistory[epoch] = cost
        if validationCostHistory[epoch] < bestCost:
            modelParametersBest = copy.deepcopy(model.state_dict())
            bestCost = validationCostHistory[epoch]   
        
    elapsedTime = time.perf_counter() - start
    if epoch % 10 == 0:
        string = "Epoch: {}/{}\t\tTraining cost: {:.2e}\t\tValidation cost: {:.2e}\nElapsed time for last epoch: {:.2f} s"
        print(string.format(epoch + 1, epochs, trainingCostHistory[epoch], validationCostHistory[epoch], elapsedTime))
    start = time.perf_counter()
    
if earlyStopping == True:
    model.load_state_dict(modelParametersBest)
print("Total elapsed time during training: {:.2f} s".format(time.perf_counter() - start0))

###############################################################################
# post processing
np.savetxt("output/trainingCost" + str(selectNN) + "_" + str(numberOfTrainingSamples) + "_" + str(weightDecay) + ".txt", trainingCostHistory)
np.savetxt("output/validationCost" + str(selectNN) + "_" + str(numberOfTrainingSamples) + "_" + str(weightDecay) + ".txt", validationCostHistory)
###############################################################################
model.eval()
if numberOfTrainingSamples == 1:
    model.train()
component = 0
x = np.linspace(0, 1, 32)
y = np.linspace(0, 1, 32)
x, y = np.meshgrid(x, y)

############################
import matplotlib
import matplotlib.font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable

#matplotlib.rcParams["figure.dpi"] = 300
#matplotlib.rcParams["axes.linewidth"] = 2.5
#from matplotlib import rc
#
#rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"], "size": 22})
#rc("text", usetex=True)
############################

def plotField(z, filename, vmin, vmax):
    fig, ax = plt.subplots(figsize=(5, 5 * 1.1))
    ax.tick_params(width=2.5, length=8)
    cp = ax.pcolormesh(x, y, z.cpu().numpy(), cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().axes.get_yaxis().set_visible(False) # or remove axes
    plt.gca().axes.get_xaxis().set_visible(False)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="7%", pad="3%")
    cb = fig.colorbar(cp, cax=cax, format="%.1f", orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    cb.locator = matplotlib.ticker.MaxNLocator(nbins=7)  # adjust nbins accoring to data
    cb.update_ticks()
    
    cb.ax.tick_params(width=2.5, length=8)
    plt.minorticks_off()
    ax.set_rasterized(True)
    fig.tight_layout()
#    plt.savefig(filename)
    plt.show()
#    plt.close()

############################
    
sample = next(iter(dataloaderTraining))
vmin = torch.min(sample[1][0][component])
vmax = torch.max(sample[1][0][component])
plotField(sample[1][0][component], "output/labelTraining" + str(selectNN) + "_" + str(numberOfTrainingSamples) + "_" + str(weightDecay) + ".pdf", vmin=vmin, vmax=vmax)

plotField(model(sample[0][0:1])[0,component].detach(), 
          "output/predictionTraining" + str(selectNN) + "_" + str(numberOfTrainingSamples) + "_" + str(weightDecay) + ".pdf", vmin=vmin, vmax=vmax)

torch.save(sample[1], "output/labelTraining" + str(selectNN) + "_" + str(numberOfTrainingSamples) + "_" + str(weightDecay) + ".pt")
torch.save(model(sample[0]).detach(), "output/predictionTraining" + str(selectNN) + "_" + str(numberOfTrainingSamples) + "_" + str(weightDecay) + ".pt")

sample = next(iter(dataloaderValidation))
vmin = torch.min(sample[1][0][component])
vmax = torch.max(sample[1][0][component])
plotField(sample[1][0][component], "output/labelValidation" + str(selectNN) + "_" + str(numberOfTrainingSamples) + "_" + str(weightDecay) + ".pdf", vmin=vmin, vmax=vmax)
plotField(model(sample[0][0:1])[0,component].detach(), 
          "output/predictionValidation" + str(selectNN) + "_" + str(numberOfTrainingSamples) + "_" + str(weightDecay) + ".pdf", vmin=vmin, vmax=vmax)

torch.save(sample[1], "output/labelValidation" + str(selectNN) + "_" + str(numberOfTrainingSamples) + "_" + str(weightDecay) + ".pt")
torch.save(model(sample[0]).detach(), "output/predictionValidation" + str(selectNN) + "_" + str(numberOfTrainingSamples) + "_" + str(weightDecay) + ".pt")


fig, ax = plt.subplots()
ax.grid()
ax.set_yscale('log')
ax.plot(validationCostHistory, 'r', label='validation')
ax.plot(trainingCostHistory, 'k', label='training')
ax.legend()
plt.show()
