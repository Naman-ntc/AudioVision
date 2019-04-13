import librosa
import pickle
from specgrams_helper import *

sampleRate = 22050
helper = SpecgramsHelper(2*sampleRate, tuple([192, 512]), 0.75, sampleRate, 1)

def writeWav(spectogram, filename = "trial.wav"):
	global helper, sampleRate
	outtrue = helper.melspecgrams_to_waves(spectogram)
	with tf.Session() as sess:
		outtrue = outtrue.eval()
	librosa.output.write_wav(filename, outtrue[0], sampleRate)

def reconstruct(spectogram):
	temp = spectogram[::4, ::4, :]
	savedtemp = temp
	import torch
	a = torch.load('model2.pth')
	# tmean = temp.mean()
	# tstd = temp.std()
	# mtemp = (temp-tmean)/tstd
	mtemp = temp
	mtmean = mtemp[:,:,0].mean()
	mtstd = mtemp[:,:,0].std()
	mtemp[:,:,0] = (mtemp[:,:,0] - mtmean)/mtstd
	temp = torch.from_numpy((mtemp).transpose(2,0,1)).to(0)
	otemp = temp
	temp = a(temp[None,:,:,:])[2][0]
	print(((temp[0,:,:] - otemp[0,:,:])**2).mean().item())
	temp = temp.detach().cpu().numpy()
	temp = temp.transpose(1,2,0)*mtstd+mtmean
	temp[:,:,1] = savedtemp[:,:,1]

	reconstructed = spectogram.copy()
	for i in range(temp.shape[0]):
		for j in range(temp.shape[1]):
			for ii in range(4):
				for jj in range(4):
					reconstructed[4*i+ii, 4*j+jj, :] = temp[i, j, :]


	return reconstructed


files = pickle.load(open("0.pkl", 'rb'))
a = files.shape[1]
b = files.shape[2]
c = files.shape[3]
reconstructed = reconstruct(files[111]).reshape(1, a, b, c)
print(files.shape, reconstructed.shape)
writeWav(files[1].reshape(1, a, b, c))
writeWav(reconstructed, filename = "temp_re.wav")

