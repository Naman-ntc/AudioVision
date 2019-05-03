import numpy as np
from imsimplemodel import *
import torch.optim as optim
import pickle
import os
import itertools
import librosa
from specgrams_helper import *

sampleRate = 22050
helper = SpecgramsHelper(2*sampleRate, tuple([192, 512]), 0.75, sampleRate, 1)

def writeWav(spectogram, filename = "trial.wav"):
				global helper, sampleRate
				outtrue = helper.melspecgrams_to_waves(spectogram)
				with tf.Session() as sess:
								outtrue = outtrue.eval()
				librosa.output.write_wav(filename, outtrue[0], sampleRate)


bigbasepath = "../../VEGAS/VEGAS/data"
trainEpoch = int(2e5)
height = 48
width = int(8 * height / 3)
gpu = "cuda:0"

weight = [1,1,1/200.]
dropLr = 15
dropBy = 2
numIter = 512
bs = 64


index=0


# mmean = mmean.detach().cpu().numpy()
# mstd = mstd.detach().cpu().numpy()

SIZE = 128
model = ImAudio(gpu=gpu, specShape = (height, width), genChannels = 2*SIZE, encChannels = 2*SIZE, latentDim = SIZE)
# mypremodel = torch.load('model2.pth').cpu()
# model.encoder = mypremodel.encoder
# model.decoder = mypremodel.decoder
model = model.to(gpu)

LR = {
				'Encoder' : 3e-4,
				'Generator' : 3e-4,
				'Image' : 1e-5,
				'more' : 3e-4,
   }

Optimizer = {
	'Encoder':optim.Adam(model.encoder.parameters(), lr=LR['Encoder']),
	'Generator':optim.Adam(model.decoder.parameters(), lr=LR['Generator']),
	'Image':optim.Adam(model.resnet.parameters(), lr=LR['Image']),
	'more':optim.Adam(model.morefc.parameters(), lr=LR['more']),
}

for i in range(trainEpoch):
	for i in range(trainEpoch):
			for jj in range(1):

				try:
					data = pickle.load(open("../../VEGAS/sound2image/data/%d.pkl"%(jj), 'rb'))
				except:
					print("shit")
					continue
				ugdata = torch.Tensor(data['spec'].copy())
				images = torch.Tensor(data['img'].transpose(0,3,1,2))
				imagemean=torch.Tensor([0.485, 0.456, 0.406])
				imagestd=torch.Tensor([0.229, 0.224, 0.225]	)
				images = (images-imagemean[None,:,None,None])/imagestd[None,:,None,None]
				# print(images.mean(), images.max())
				labels = torch.LongTensor(data['labels'])#torch.LongTensor(labels)
				data = torch.Tensor(data['spec'])
				data = np.transpose(data, [0, 3, 1, 2])
				data = torch.Tensor(data)[:,:,::4,::4]
				mstd = [0 for _ in range(data.shape[0])]
				mmean = [0 for _ in range(data.shape[0])]
				for iij in range(data.shape[0]):
				                mmean[iij] = data[iij].mean().detach().cpu().numpy()
				                mstd[iij] = data[iij].std().detach().cpu().numpy()
				                data[iij] = (data[iij] - data[iij].mean()) / data[iij].std()

				numIter = data.shape[0] // bs
				print_losses = [0, 0, 0]
				for j in range(numIter):
								#print("lala1")

								currentData = data[j*bs:(j + 1)*bs].to(gpu)
								currentImages = (images[j*bs:(j + 1)*bs].to(gpu))
								#print("lala2")

								temp = list(model(currentData[:,:1,:,:], currentImages))
								temp+=[currentData[:,:1,:,:]]
								losses = vae_loss(*temp)
								loss = sum([x*y for x,y in zip(list(losses), weight)])
								loss.backward()

								if i%25 == 24:


												# temp2 = np.zeros(tuple([ temp[2][0].shape[1], temp[2][0].shape[2], 2]))
												# reconstructedMean = temp[2]
												# # temp2 = temp2.detach().cpu().numpy()
												# temp2[:,:,0] = reconstructedMean[0,0,:,:].detach().cpu().numpy()
												# temp2 = temp2*mstd[j]+mmean[j]
												# temp2[:,:,1] = currentData[0,1,:,:]
												# reconstructed = np.zeros(tuple([192,512,2]))
												# for iii in range(temp2.shape[0]):
												# 				for jjj in range(temp2.shape[1]):
												# 								for ii in range(4):
												# 												for jj in range(4):
												# 																reconstructed[4*iii+ii, 4*jjj+jj, :] = temp2[iii, jjj, :]

												# reconstructed = reconstructed.reshape((1, data.shape[2]*4, data.shape[3]*4, data.shape[1]))
												# reconstructed[0,:,:,1] = ugdata[j*bs,:,:,1]
												# writeWav(reconstructed.astype(np.float32), filename = "wavdumps_without/aud_re"+str(j)+".wav")

												temp2 = np.zeros(tuple([ temp[3][0].shape[1], temp[3][0].shape[2], 2]))
												reconstructedMean = temp[3]
												# temp2 = temp2.detach().cpu().numpy()
												temp2[:,:,0] = reconstructedMean[0,0,:,:].detach().cpu().numpy()
												temp2 = temp2*mstd[j]+mmean[j]
												temp2[:,:,1] = currentData[0,1,:,:]
												reconstructed = np.zeros(tuple([192,512,2]))
												for iii in range(temp2.shape[0]):
																for jjj in range(temp2.shape[1]):
																				for ii in range(4):
																								for jj in range(4):
																												reconstructed[4*iii+ii, 4*jjj+jj, :] = temp2[iii, jjj, :]

												reconstructed = reconstructed.reshape((1, data.shape[2]*4, data.shape[3]*4, data.shape[1]))
												reconstructed[0,:,:,1] = ugdata[j*bs,:,:,1]
												writeWav(reconstructed.astype(np.float32), filename = "wavdumps_without/im_re"+str(j)+".wav")


												temp2 = np.zeros(tuple([ temp[3][0].shape[1], temp[3][0].shape[2], 2]))
												reconstructedMean = currentData
												# temp2 = temp2.detach().cpu().numpy()
												temp2[:,:,0] = reconstructedMean[0,0,:,:].detach().cpu().numpy()
												temp2 = temp2*mstd[j]+mmean[j]
												temp2[:,:,1] = currentData[0,1,:,:]
												reconstructed = np.zeros(tuple([192,512,2]))
												for iii in range(temp2.shape[0]):
																for jjj in range(temp2.shape[1]):
																				for ii in range(4):
																								for jj in range(4):
																												reconstructed[4*iii+ii, 4*jjj+jj, :] = temp2[iii, jjj, :]

												reconstructed = reconstructed.reshape((1, data.shape[2]*4, data.shape[3]*4, data.shape[1]))
												reconstructed[0,:,:,1] = ugdata[j*bs,:,:,1]
												writeWav(reconstructed.astype(np.float32), filename = "wavdumps_without/true"+str(j)+".wav")


								#print(currentData.shape, reconstructedMean.shape)
								print_losses[0] += 0#losses[0].item()*weight[0]
								print_losses[1] += losses[1].item()*weight[1]
								print_losses[2] += 0#losses[2].item()*weight[2]
								#((currentData - reconstructedMean)**2).mean().item()

								for _,myoptim in Optimizer.items():
												myoptim.step()
												myoptim.zero_grad()

				print(i, print_losses[0]/numIter, "\t", print_losses[1]/numIter, "\t", print_losses[2] / numIter)

			if i %dropLr == dropLr - 1:
					LR['Encoder']/= dropBy
					LR['Generator']/= dropBy
			if True:
					torch.save(model, 'modelnoaud.pth')



