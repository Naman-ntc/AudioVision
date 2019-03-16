import os
import torch
#from opts import opts
import pickle
import torch.optim as optim
from progress.bar import Bar
import torch.nn.functional as F
from utils import AverageMeter, ensure_dir
from data import PickleLoader
from models import AudioGenerator, AudioEncoder, AudioDiscriminator, ImageEncoder

import librosa
from specgrams_helper import *

def temp(epoch):
        i=2
        j=7
        sampleRate = 22050
        Path = os.path.join(BasePath, 'subdata' + str(i+1), 'wav', str(j)+'.pkl')
        specs = torch.from_numpy(pickle.load(open(Path, 'rb')).transpose(0,3,1,2)).to(gpu)[:5]
        if epoch!='gt':
                specs = Model['Generator'](Model['Encoder'](specs)[:,:,0,0])
        helper = SpecgramsHelper(2*sampleRate, tuple([192, 512]), 0.75, sampleRate, 1)
        out = helper.melspecgrams_to_waves(specs.cpu().detach().numpy().transpose(0,2,3,1))
        import tensorflow as tf
        with tf.Session() as sess:
                out = out.eval()
        #os.mkdir()#pickle.dump(open('dumps/'+str(epoch)+'.pkl', 'rb'))
        ensure_dir((os.path.join(EXP_BASE, EXP_DIR, 'dumps', str(epoch))))
        for i in range(out.shape[0]):
                librosa.output.write_wav(os.path.join(EXP_BASE, EXP_DIR, 'dumps', str(epoch), str(i)+'.wav'), out[i], sampleRate)#
        return

#opts = opts().parse()
gpu = 0
EXP_BASE = 'exp'
EXP_DIR = 'Adam-Small-Latent-Batchnorm-Reorder'
batchSize = 32
LR = {'Encoder' : 1e-3, 'Generator' : 1e-3}
Model = {'Encoder':AudioEncoder().to(gpu), 'Generator':AudioGenerator().to(gpu)}
Optimizer = {'Encoder':optim.Adam(Model['Encoder'].parameters(), lr=LR['Encoder']), 'Generator':optim.Adam(Model['Generator'].parameters(), lr=LR['Generator'])}

BasePath = '../VEGAS/VEGAS/data/'

ensure_dir((os.path.join(EXP_BASE, EXP_DIR)))
temp('gt')
file = os.path.join(EXP_BASE, EXP_DIR, 'log.txt')#, 'a')
for epoch in range(1300):
        for i in range(4):
                for j in range(10):
                        i=2
                        j=7
                        Path = os.path.join(BasePath, 'subdata' + str(i+1), 'wav', str(j)+'.pkl')
                        DataLoader = PickleLoader(pickle.load(open(Path, 'rb')), batchSize)
                        Loss = AverageMeter()
                        nIters = len(DataLoader)
                        bar = Bar('==>', max=nIters)
                        for batch_idx,specs in enumerate(DataLoader):
                                specs = specs.to(gpu)
                                constructed = Model['Generator'](Model['Encoder'](specs)[:,:,0,0])
                                loss = 0
                                #loss += F.mse_loss(constructed, specs)
                                loss += F.l1_loss(constructed, specs)
                                loss.backward()
                                Loss.update(loss.item(), specs.shape[0])
                                for _,myoptim in Optimizer.items():
                                        myoptim.step()
                                        myoptim.zero_grad()
                                Bar.suffix = ' Epoch: [{0}][{1}][{2}][{3}/{4}]| Total: {total:} | ETA: {eta:} | Loss: {loss.avg:.6f} ({loss.val:.6f})'.format(epoch, i, j, batch_idx+1, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss)
                                bar.next()
                        bar.finish()
        if (epoch+1)%10==0:
                wfile = open(file, 'a')
                wfile.write('Epoch = %d, Loss = %f\n'%(epoch, Loss.avg))
                wfile.close()
        if (epoch+1)%400==0:
                for i,param_group in enumerate(Optimizer['Encoder'].param_groups):
                        param_group['lr'] *= 0.1
                for i,param_group in enumerate(Optimizer['Generator'].param_groups):
                        param_group['lr'] *= .1
        if (epoch+1)%10==0:
                temp(epoch+1)
