import torch
import torch.optim as optim
from opts import opts
from models import AudioVAE,vae_loss
from trainer import Trainer

opts = opts().parse()

Model = AudioVAE(
		opts.latentDim, 
		opts.encChannels, 
		opts.genChannels, 
		opts.specRatio, 
		opts.specShape, 
		opts.leakyRelu, 
		opts.pixelNorm, 
		opts.normalize_latents, 
		opts.gpu
	)
Model = Model.to(opts.gpu)

LR = {
	'Encoder' : opts.encLR, 
	'Generator' : opts.genLR,
   }

Optimizer = {
		'Encoder':getattr(optim, opts.enc_optimizer_type)(Model.encoder.parameters(), lr=LR['Encoder']), 
		'Generator':getattr(optim, opts.gen_optimizer_type)(Model.decoder.parameters(), lr=LR['Generator'])##Nomenclature TODO
	}

LossFunc = vae_loss
trainer = Trainer(Model,Optimizer,LossFunc,opts)
trainer.train(1,100)
#BasePath = opts.basepath