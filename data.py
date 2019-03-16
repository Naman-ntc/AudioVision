import pickle
import torch.utils.data as data


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
	def __init__(self, pickle):
		super(PickleDataset, self).__init__()
		self.pickle = pickle

	def __getitem__(self, index):
		return self.pickle[index].transpose(2,0,1)

	def __len__(self):
		return 8#self.pickle.shape[0]

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

