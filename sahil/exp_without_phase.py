import numpy as np
from simplemodel_without_phase import *
import torch.optim as optim
import pickle
import os

trainEpoch = int(2e5)
height = 48
width = int(8 * height / 3)
gpu = "cuda:1"

weight = 0.0003
dropLr = 600
dropBy = 2
numIter = 512
bs = 128

EXP_BASE = "."
EXP_DIR = "trial"

def temp(epoch):
	i=int(torch.randint(4,()))
	j=int(torch.randint(10,()))
	i=2
	j=7
	sampleRate = 22050
	Path = os.path.join(BasePath, 'subdata' + str(i+1), 'wav', str(j)+'.pkl')
	specs = torch.from_numpy(pickle.load(open(Path, 'rb')).transpose(0,3,1,2)).to(gpu)
	N = 20
	specs = specs[np.random.choice(specs.shape[0], N)]
	mean,sigma,reconstructed = Model(specs)
	losses = vae_loss(mean, sigma, reconstructed, specs, weight)
	print(losses[0].item()/N, losses[1].item()/N)	

	helper = SpecgramsHelper(2*sampleRate, tuple([192, 512]), 0.75, sampleRate, 1)
	outtrue = helper.melspecgrams_to_waves(specs.cpu().detach().numpy().transpose(0,2,3,1))
	#outre = helper.melspecgrams_to_waves((mystats[1][None,:,:,:]*reconstructed.cpu().detach().numpy()+mystats[0][None,:,:,:]).transpose(0,2,3,1))
	outre = helper.melspecgrams_to_waves((reconstructed.cpu().detach().numpy()).transpose(0,2,3,1))
	import tensorflow as tf
	with tf.Session() as sess:
		outtrue = outtrue.eval()
		outre = outre.eval()
	ensure_dir((os.path.join(EXP_BASE, EXP_DIR, 'dumps', str(epoch))))
	for k in range(outtrue.shape[0]):
		librosa.output.write_wav(os.path.join(EXP_BASE, EXP_DIR, 'dumps', str(epoch), str(i)+"_"+str(j)+"_"+str(k)+'.wav'), outre[k], sampleRate)#
	ensure_dir((os.path.join(EXP_BASE, EXP_DIR, 'dumps', str('gt'))))
	for k in range(outre.shape[0]):
		librosa.output.write_wav(os.path.join(EXP_BASE, EXP_DIR, 'dumps', str(epoch), str(i)+"_"+str(j)+"_"+str(k)+'_gt'+'.wav'), outtrue[k], sampleRate)#	
	return


index=0

odata = torch.Tensor(np.random.randn(bs * numIter, 2, height, width)).to(gpu)
data = pickle.load(open("0.pkl", 'rb'))
data = np.transpose(data, 	[0, 3, 1, 2])[:,:,::4,::4]
# data = data[:,0:1,:,:]
data = torch.Tensor(data).to(gpu)#[:256]
numIter = data.shape[0] // bs
print(data.shape,odata.shape)

for i in range(data.shape[0]):
	data[i] = data[i] - data[i].mean().to(gpu)
	data[i] = data[i] / np.abs(data[i]).max().to(gpu)

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
	temp(i)

	if i %dropLr == dropLr - 1:
		LR['Encoder']/= dropBy
		LR['Generator']/= dropBy
	if i%2000:
		torch.save(model, 'model2.pth')
