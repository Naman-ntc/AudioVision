import os
import utils as U
import configargparse

class opts():
	"""docstring for opts"""
	def __init__(self):
		super(opts, self).__init__()
		self.parser = configargparse.ArgParser(default_config_files=['default.yaml'])

	def init(self):
		self.parser.add('-DEBUG', action='store_true', help='To run in debug mode!!')

		self.parser.add('--expDir', help='Experiment-Directory')
		self.parser.add('--expID', help='Experiment-ID')

		self.parser.add('--gpu', type=int)
		self.parser.add('--batch_size', type=int, help='Experiment-ID')
		self.parser.add('--mini_batch_count', type=int, help='After how many mini batches to run backprop')		



		##
		self.parser.add('--encLR', type=float, help='Experiment-ID')
		self.parser.add('--genLR', type=float, help='Experiment-ID')
		self.parser.add('--enc_optimizer_type', type=str, help='Experiment-ID')
		self.parser.add('--gen_optimizer_type', type=str, help='Experiment-ID')

		self.parser.add('--saveInterval', type=int, help='After how many train epochs to save model')
		self.parser.add('--dropInterval', type=int, help='After how many train epochs to save model')
		self.parser.add('--dumpInterval', type=int, help='After how many train epochs to save model')
		self.parser.add('--dropLR', type=int, help='Drop LR after how many epochs')
		self.parser.add('--dropMag', type=float, help='Drop LR magnitude')
		
		self.parser.add('--weight', type=float, help='Experiment-ID')
		
		self.parser.add('--basepath', type=str, help='Experiment-ID')
		#Model
		self.parser.add('--latentDim', type=int, help='Experiment-ID')
		self.parser.add('--encChannels', type=int, help='Experiment-ID')
		self.parser.add('--genChannels', type=int, help='Experiment-ID')
		self.parser.add('--specRatio', type=str, help='Experiment-ID')
		self.parser.add('--specShape', type=str, help='Experiment-ID')
		self.parser.add('--leakyRelu', type=int, help='Experiment-ID')
		self.parser.add('--pixelNorm', type=int, help='Experiment-ID')
		self.parser.add('--normalize_latents', type=int, help='Experiment-ID')
		self.parser.add('--normalize_data', type=int, help='Experiment-ID')


	def parse(self):
		self.init()
		self.opt = self.parser.parse_args()
		self.opt.saveDir = os.path.join(self.opt.expDir, self.opt.expID)
		self.opt.dumpDir = os.path.join(self.opt.saveDir, 'dumps')
		self.opt.logFile = os.path.join(self.opt.saveDir, 'log.txt')
		#os.path.join(EXP_BASE, EXP_DIR, 'log.txt')
		U.ensure_dir(self.opt.saveDir)
		U.ensure_dir(self.opt.dumpDir)
		self.opt.specRatio = [int(x) for x in self.opt.specRatio.split(',')]
		self.opt.specShape = [int(x) for x in self.opt.specShape.split(',')]
		file_name = os.path.join(self.opt.saveDir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('==> Args:\n')
			args = dict((name, getattr(self.opt, name)) for name in dir(self.opt) if not name.startswith('_'))
			for k, v in sorted(args.items()):
				opt_file.write("%s: %s\n"%(str(k), str(v)))

		return self.opt