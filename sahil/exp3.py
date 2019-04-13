import numpy as np 
from models3 import *
import torch.optim as optim
import pickle

trainEpoch = int(2e5)
height = 192
width = int(8 * height / 3)
gpu = 0

weight = 0.001
dropLr = 200
dropBy = 2
numIter = 1
bs = 32

# data = torch.Tensor(np.random.randn(bs * numIter, 2, height, width)).to(gpu)
data = pickle.load(open("0.pkl", 'rb'))
data = np.transpose(data, 	[0, 3, 1, 2])
smallData = data[:,:,::4,::4]
data = torch.Tensor(data)[:32]
numIter = data.shape[0] // bs
print(data.shape[0])

for i in range(data.shape[0]):
	data[i] = (data[i] - data[i].mean()) / data[i].std()

model = AudioVAE(gpu=gpu, specShape = (height, width), encChannels = 128, genChannels = 128, latentDim = 128).to(gpu)

LR = {
	'Encoder' : 3e-4, 
	'Generator' : 3e-4,
   }

Optimizer = {
		'Encoder':optim.Adam(model.encoder.parameters(), lr=LR['Encoder']), 
		'Generator':optim.Adam(model.decoder.parameters(), lr=LR['Generator'])
	}

for i in range(trainEpoch):

	print_losses = [0, 0, 0]
	for j in range(numIter):
		currentData = data[j*bs:(j + 1)*bs].to(gpu)
		# currentData = data[j*bs:(j + 1)*bs]
		mean, sigma, reconstructedMean, reconstructedSigma = model(currentData)

		losses = vae_loss(mean, sigma, reconstructedMean, reconstructedSigma, currentData, weight)
		loss = sum(list(losses))
		loss.backward()

		print_losses[0] += losses[0].item()
		print_losses[1] += losses[1].item()
		print_losses[2] += ((currentData - reconstructedMean)**2).mean().item()

		for _,myoptim in Optimizer.items():
			myoptim.step()
			myoptim.zero_grad()

	print(i, print_losses[0]/numIter / (weight - 1e-8), "\t", print_losses[1]/numIter, "\t", print_losses[2] / numIter)
		
	if i %dropLr == dropLr - 1:
		LR['Encoder']/= dropBy
		LR['Generator']/= dropBy