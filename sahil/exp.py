import numpy as np 
from models import *
import torch.optim as optim

trainEpoch = int(2e5)
height = 6
width = int(8 * height / 3)

weight = 0.001
dropLr = 200
dropBy = 2
numIter = 512
bs = 32

data = torch.Tensor(np.random.randn(bs * numIter, 2, height, width)).to(0)

model = AudioVAE(gpu=0, specShape = (height, width), encChannels = 64, latentDim = 64).to(0)

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
		currentData = data[j*bs:(j + 1)*bs]
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