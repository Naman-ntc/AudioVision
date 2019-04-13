import numpy as np
from models import *
# from models4 import *
import torch.optim as optim
import pickle

trainEpoch = int(2e5)
height = 48
width = int(8 * height / 3)
gpu = "cuda:1"
SIZE = 128

weight = 0.0003
dropLr = 600
dropBy = 2
numIter = 512
bs = 128
lr = 3e-4


index=0


model = AudioVAE(gpu=gpu, specShape = (height, width), genChannels = 2*SIZE, encChannels = 2*SIZE, latentDim = SIZE)
model = model.to(gpu)
print("loaded model")

odata = torch.Tensor(np.random.randn(bs * numIter, 2, height, width)).to(gpu)
data = pickle.load(open("0.pkl", 'rb'))
data = np.transpose(data, 	[0, 3, 1, 2])[:,:,::4,::4]
# data = data[:,0:1,:,:]
data = np.random.randn(128, 2, 48, 128)
data = torch.Tensor(data).to(gpu)[:128]
numIter = data.shape[0] // bs
print(data.shape,odata.shape)

# for i in range(data.shape[0]):
# 	data[i, 0] = (data[i, 0] - data[i, 0].mean()) / data[i, 0].std()
# 	data[i, 1] = (data[i, 1] - data[i, 1].mean()) / data[i, 1].std()


LR = {
	'Encoder' : lr,
	'Generator' : lr,
   }

Optimizer = {
		'Encoder':optim.Adam(model.encoder.parameters(), lr=LR['Encoder']),
		'Generator':optim.Adam(model.decoder.parameters(), lr=LR['Generator'])
	}

for i in range(trainEpoch):

	print_losses = [0, 0, 0, 0]
	for j in range(numIter):
		currentData = data[j*bs:(j + 1)*bs]
		mean, sigma, reconstructedMean, reconstructedSigma = model(currentData)

		losses = vae_loss(mean, sigma, reconstructedMean, reconstructedSigma, currentData, weight)
		loss = sum(list(losses))
		loss.backward()

		print_losses[0] += losses[0].item()
		print_losses[1] += losses[1].item()
		print_losses[2] += ((currentData[:, 0, :, :] - reconstructedMean[:, 0, :, :])**2).mean().item()
		print_losses[3] += ((currentData[:, 1, :, :] - reconstructedMean[:, 1, :, :])**2).mean().item()

		for _,myoptim in Optimizer.items():
			myoptim.step()
			myoptim.zero_grad()

	print(i, print_losses[0]/numIter / (weight), "\t", print_losses[1]/numIter, "\t", print_losses[2] / numIter, "\t", print_losses[3] / numIter)

	if i %dropLr == dropLr - 1:
		LR['Encoder']/= dropBy
		LR['Generator']/= dropBy
	if i%2000:
		torch.save(model, 'model2.pth')
