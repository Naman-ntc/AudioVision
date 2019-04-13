import os
import torch
import pickle
import itertools
from progress.bar import Bar
from utils import AverageMeter
from data import PickleLoader,PickleDataset

class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, Model, Optimizer, LossFunc, opts):
		super(Trainer, self).__init__()
		self.model = Model
		self.optimizer = Optimizer
		self.LossFunc = LossFunc
		self.opts = opts
		self.gpu = self.opts.gpu

	def train(self, startepoch, endepoch):
		for epoch in range(startepoch, endepoch+1):
			self.dump(epoch)
			train = self._epoch(epoch)

			Writer = open(self.File, 'a')
			Writer.write(train + '\n')
			Writer.close()

			if epoch%self.opts.dumpInterval==0:
				self.dump(epoch)

			if epoch%self.opts.saveInterval==0:
			        state = {
			                'epoch': epoch+1,
			                'model_state': self.model.state_dict(),
			        }
			        path = os.path.join(self.opts.saveDir, 'model_{}.pth'.format(epoch))
			        torch.save(state, path)

			if epoch%self.opts.dropInterval==0:
				for i,param_group in enumerate(self.optimizer['Encoder'].param_groups):
					param_group['lr'] *= self.opts.dropMag
				for i,param_group in enumerate(self.optimizer['Generator'].param_groups):
					param_group['lr'] *= self.opts.dropMag

	def initepoch(self):
	    self.loss = AverageMeter(),AverageMeter()
	    self.loss[0].reset()
	    self.loss[1].reset()

	def backprop(self):
		for myoptim in self.optimizer.values():
			myoptim.step()
			myoptim.zero_grad()

	def _epoch(self, epoch):
		self.initepoch()
		for i,j in itertools.product(range(4), range(10)):
			model = self.model.to(self.gpu)
			
			if self.opts.DEBUG:
				i,j = 2,7
			Path = os.path.join(self.opts.basepath, 'subdata' + str(i+1), 'wav', str(j)+'.pkl')
			DataLoader = PickleLoader(pickle.load(open(Path, 'rb')), self.opts)
			
			nIters = len(DataLoader)
			bar = Bar('==>', max=nIters)
			for batch_idx,specs in enumerate(DataLoader):
				specs = specs.to(self.gpu)
				mean, sigma, reconstructed = self.model(specs)
				if self.opts.DEBUG:
					print([x.item() for x in [reconstructed.mean(),reconstructed.std(),reconstructed.max(),reconstructed.min()]])
				losses = self.LossFunc(mean, sigma, reconstructed, specs, self.opts.weight)
				loss = sum(list(losses))
				loss.backward()
				self.loss[0].update(losses[0].item(), specs.shape[0])
				self.loss[1].update(losses[1].item(), specs.shape[0])
				if (epoch+1)%self.opts.mini_batch_count==0:
					self.backprop()
				Bar.suffix = ' Epoch: [{0}][{1}][{2}][{3}/{4}]| Total: {total:} | ETA: {eta:} | Latent-Loss: {loss[0].avg:.6f} ({loss[0].val:.6f}) | Reconstruction-Loss: {loss[1].avg:.6f} ({loss[1].val:.6f}) '.format(epoch, i, j, batch_idx+1, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=self.loss)
				bar.next()
			bar.finish()
		return 'Latent : {:8f}, Reconstruction : {:8f} '.format(self.loss[0].avg, self.loss[1].avg)

	def dump(self, epoch):
		i=2 if self.opts.DEBUG else int(torch.randint(4,())) 
		j=2 if self.opts.DEBUG else int(torch.randint(10,()))

		sampleRate = 22050
		Path = os.path.join(self.opts.basepath, 'subdata' + str(i+1), 'wav', str(j)+'.pkl')
		Dataset = PickleDataset(pickle.load(open(Path, 'rb')), self.opts)		
		specs = Dataset.getBatch(20).to(self.gpu)
		mean,sigma,reconstructed = self.model(specs)
		outtrue, outre = Dataset.reconstruct(specs,reconstructed)
		# losses = vae_loss(mean, sigma, reconstructed, specs, weight)
		# print(losses[0].item()/N, losses[1].item()/N)	
		mysaveDir = os.path.join(self.opts.dumpDir, str(epoch))
		ensure_dir(mysaveDir)
		for k in range(outtrue.shape[0]):
			librosa.output.write_wav(os.path.join(mysaveDir, str(epoch), str(i)+"_"+str(j)+"_"+str(k)+'.wav'), outre[k], sampleRate)#
		for k in range(outre.shape[0]):
			librosa.output.write_wav(os.path.join(mysaveDir, str(epoch), str(i)+"_"+str(j)+"_"+str(k)+'_gt'+'.wav'), outtrue[k], sampleRate)#
		return
