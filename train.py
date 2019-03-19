import os
import torch
import pickle
import itertools
import torch.optim as optim
from progress.bar import Bar
import torch.nn.functional as F
from utils import AverageMeter, ensure_dir
from data import PickleLoader
from models import *#AudioGenerator, AudioEncoder, ImageEncoder
from random import randint

import librosa
from specgrams_helper import *

def temp(epoch):
	i=int(torch.randint(4,()))
	j=int(torch.randint(10,()))
	#i=2
	#j=7
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
	outre = helper.melspecgrams_to_waves(reconstructed.cpu().detach().numpy().transpose(0,2,3,1))
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

#opts = opts().parse()
gpu = 0
EXP_BASE = 'exp'
EXP_DIR = 'VAE-complete'
batchSize = 32

weight = 1

Model = AudioVAE(gpu=gpu).to(gpu)

LR = {
	'Encoder' : 1e-4, 
	'Generator' : 1e-4,
   }

Optimizer = {
		'Encoder':optim.Adam(Model.encoder.parameters(), lr=LR['Encoder']), 
		'Generator':optim.Adam(Model.decoder.parameters(), lr=LR['Generator'])
	}
BasePath = '../VEGAS/VEGAS/data/'

ensure_dir((os.path.join(EXP_BASE, EXP_DIR)))
file = os.path.join(EXP_BASE, EXP_DIR, 'log.txt')

savedspecs = None
for epoch in range(1300):
	for i,j in itertools.product(range(4), range(10)):
		#i=2
		#j=7
		Path = os.path.join(BasePath, 'subdata' + str(i+1), 'wav', str(j)+'.pkl')
		try:
			DataLoader = PickleLoader(pickle.load(open(Path, 'rb')), batchSize)
		except:
			print("Ignoring i=%d, j=%d"%(i,j))
			continue
		Losses = AverageMeter(),AverageMeter()
		nIters = len(DataLoader)
		bar = Bar('==>', max=nIters)
		for batch_idx,specs in enumerate(DataLoader):
			specs = specs.to(gpu)
			mean, sigma, reconstructed = Model(specs)
			losses = vae_loss(mean, sigma, reconstructed, specs, weight)
			loss = sum(list(losses))
			loss.backward()
			Losses[0].update(losses[0].item(), specs.shape[0])
			Losses[1].update(losses[1].item(), specs.shape[0])
			for _,myoptim in Optimizer.items():
				myoptim.step()
				myoptim.zero_grad()
			Bar.suffix = ' Epoch: [{0}][{1}][{2}][{3}/{4}]| Total: {total:} | ETA: {eta:} | Latent-Loss: {loss[0].avg:.6f} ({loss[0].val:.6f}) | Reconstruction-Loss: {loss[1].avg:.6f} ({loss[1].val:.6f}) '.format(epoch, i, j, batch_idx+1, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Losses)
			bar.next()
		bar.finish()
		if (epoch+1)%1==0:
			wfile = open(file, 'a')
			wfile.write('Epoch = %d, i=%d, j=%d, Latent-Loss = %f, Reconstruction-Loss = %f\n'%(epoch, i, j, Losses[0].avg, Losses[1].avg))
			wfile.close()
	if (epoch+1)%12==0:
		for i,param_group in enumerate(Optimizer['Encoder'].param_groups):
			param_group['lr'] *= 0.2
		for i,param_group in enumerate(Optimizer['Generator'].param_groups):
			param_group['lr'] *= 0.2
	if (epoch+1)%1==0:
		temp(epoch)