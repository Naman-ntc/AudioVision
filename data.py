import pickle
import torch.utils.data as data

import numpy as np
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
	def __init__(self, mypickle):
		super(PickleDataset, self).__init__()
		self.mypickle = mypickle.transpose(0,3,1,2)
		self.stats = tuple(x.astype(np.float) for x in pickle.load(open('stats.pkl', 'rb')))
		zeros = np.where(self.stats[1]==0)
		self.stats[0][zeros]=0
		self.stats[1][zeros]=1

		#self.randomdata = np.random.randn(5,2,192,512)

	def __getitem__(self, index):
		#return self.randomdata[index].astype(np.float32)
		#temp = (self.mypickle[index] - self.stats[0])/self.stats[1]
		#return temp.astype(np.float32)
		return self.mypickle[index]

	def __len__(self):
		return 5#self.mypickle.shape[0]

def PickleLoader(pickle, bs):
	return data.DataLoader(
			dataset = PickleDataset(pickle),
			batch_size = bs,
			shuffle = True,
			pin_memory = True,
		)

def AudioDataLoader(opts, split):
	return data.DataLoader(
				dataset = VEGAS(opts, split),
				batch_size = opts.data_loader_size,
				shuffle = True,
				pin_memory = True,
	)

