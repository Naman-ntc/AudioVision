import pickle
import torch.utils.data as data
import torch
import librosa
import numpy as np
import tensorflow as tf
from specgrams_helper import *

# class VEGAS(data.Dataset):
# 	"""docstring for VEGAS"""
# 	def __init__(self, opts, split):
# 		super(VEGAS, self).__init__()
# 		self.opts = opts
# 		self.split = split
# 		self.count = 0
# 		self.nloads = 40

# 	def __getitem__(self, index):


# 	def __len__(self):
# 		return len(self.files)

# 	def reset(self):
# 		self.count = 0



class PickleDataset(data.Dataset):
	"""docstring for PickleDataset"""
	def __init__(self, mypickle, opts):
		super(PickleDataset, self).__init__()
		self.mypickle = mypickle.transpose(0,3,1,2)
		self.stats = tuple(x.astype(np.float) for x in pickle.load(open('stats.pkl', 'rb')))
		zeros = np.where(self.stats[1]==0)
		self.stats[0][zeros]=0
		self.stats[1][zeros]=1
		self.opts = opts

		self.inittf()

	def __getitem__(self, index):
		if self.opts.normalize_data:
			self.__getitem__ = lambda index : ((self.mypickle[index] - self.stats[0])/self.stats[1]).astype(np.float32)
		else:
			self.__getitem__ = lambda index : self.mypickle[index]
		return self.__getitem__(index)

	def __len__(self):
		if self.opts.DEBUG:
			return 5
		else:
			return self.mypickle.shape[0]
	
	def getBatch(self,size):
		if self.opts.DEBUG:
			return self.mypickle[:5]
		indices = torch.randint(0, len(self.mypickle), (size,)).numpy().astype(int)
		return torch.from_numpy(self.mypickle[indices])

	def inittf(self):
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

	def reconstruct(self, gtspecs, constr):
		sampleRate = 22050
		helper = SpecgramsHelper(2*sampleRate, (self.opts.specShape), 0.75, sampleRate, 1)
		if self.opts.normalize_data:
			gtout = helper.melspecgrams_to_waves((self.stats[1][None,:,:,:]*gtspecs.cpu().detach().numpy()+self.stats[0][None,:,:,:]).transpose(0,2,3,1))
		else:
			gtout = helper.melspecgrams_to_waves(gtspecs.cpu().detach().numpy().transpose(0,2,3,1))
		if self.opts.normalize_data:
			constrout = helper.melspecgrams_to_waves((self.stats[1][None,:,:,:]*constr.cpu().detach().numpy()+self.stats[0][None,:,:,:]).transpose(0,2,3,1))
		else:
			constrout = helper.melspecgrams_to_waves(constr.cpu().detach().numpy().transpose(0,2,3,1))
		self.sess.run((gtmy,gtout))
		return gtout,constrout


def PickleLoader(pickle, opts):
	return data.DataLoader(
			dataset = PickleDataset(pickle, opts),
			batch_size = opts.batch_size,
			shuffle = True,
			pin_memory = True,
		)

