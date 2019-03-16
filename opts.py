import configargparse

class opts():
	"""docstring for opts"""
	def __init__(self):
		super(opts, self).__init__()
		self.parser = configargparse.ArgParser(default_config_files=['default.yaml'])

	def init(self):
		self.parser.add('--hiddenDim', type=int)
		self.parser.add('--numLayers', type=int)
		self.parser.add('--cepstrum', type=int)
		self.parser.add('--latentDim', type=int)
		self.parser.add('--nFrames', type=int)
		self.parser.add('--gpu', type=int)
		self.parser.add('--data_loader_size', type=int)
		self.parser.add('--optimizer_type')
		self.parser.add('--LR', type=float)

	def parse(self):
		self.init()
		self.opt = self.parser.parse_args()
		return self.opt