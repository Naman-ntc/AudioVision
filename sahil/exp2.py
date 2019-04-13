import numpy as np
from simplemodel import *
import torch.optim as optim
import pickle

trainEpoch = int(2e5)
height = 48
width = int(8 * height / 3)
gpu = "cuda:1"

weight = 0.0003
dropLr = 600
dropBy = 2
numIter = 512
bs = 128


index=0

odata = torch.Tensor(np.random.randn(bs * numIter, 2, height, width)).to(gpu)
data = pickle.load(open("0.pkl", 'rb'))
data = np.transpose(data, 	[0, 3, 1, 2])[:,:,::4,::4]
data = data[:,0:1,:,:]
data = torch.Tensor(data).to(gpu)#[:256]
numIter = data.shape[0] // bs
print(data.shape,odata.shape)

for i in range(data.shape[0]):
	data[i] = (data[i] - data[i].mean()) / data[i].std()

SIZE = 128
model = AudioVAE(gpu=gpu, specShape = (height, width), genChannels = 2*SIZE, encChannels = 2*SIZE, latentDim = SIZE)
model = model.to(gpu)

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

		losses = vae_loss(mean, sigma, reconstructedMean, reconstructedSigma, currentData, index, weight)
		loss = sum(list(losses))
		loss.backward()

		print_losses[0] += losses[0].item()
		print_losses[1] += losses[1].item()
		print_losses[2] += ((currentData - reconstructedMean)**2).mean().item()

		for _,myoptim in Optimizer.items():
			myoptim.step()
			myoptim.zero_grad()

	print(i, print_losses[0]/numIter / (weight), "\t", print_losses[1]/numIter, "\t", print_losses[2] / numIter)

	if i %dropLr == dropLr - 1:
		LR['Encoder']/= dropBy
		LR['Generator']/= dropBy
	if i%2000:
		torch.save(model, 'model2.pth')
