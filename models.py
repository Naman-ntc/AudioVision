import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import modules.audiomodules as A

class AudioGenerator(nn.Module):
        """docstring for AudioGenerator"""
        def __init__(self, latentDim=16, genChannels=256, specRatio=(3,8), specShape=(192,512), pixelNorm=False, leakyRelu=True, normalize_latents=False):
                super(AudioGenerator, self).__init__()
                self.latentDim = latentDim
                self.genChannels = genChannels
                self.specRatio = specRatio
                self.specShape = specShape
                self.pixelNorm = pixelNorm
                self.leakyRelu = leakyRelu
                self.normalize_latents = normalize_latents
                self.eps = 1e-8

                self.initblock = A.GenInitBlock(latentDim, genChannels, specRatio, pixelNorm, leakyRelu)
                self.blocks = nn.ModuleList([
                        A.GenBlock(genChannels, genChannels, pixelNorm, leakyRelu, deConv=True),
                        A.GenBlock(genChannels, genChannels, pixelNorm, leakyRelu, deConv=False),
                        A.GenBlock(genChannels, genChannels, pixelNorm, leakyRelu, deConv=True),
                        A.GenBlock(genChannels, genChannels, pixelNorm, leakyRelu, deConv=False),
                        A.GenBlock(genChannels, genChannels//2, pixelNorm, leakyRelu, deConv=True),
						A.GenBlock(genChannels//2, genChannels//2, pixelNorm, leakyRelu, deConv=False),
                        A.GenBlock(genChannels//2, genChannels//4, pixelNorm, leakyRelu, deConv=True),
						A.GenBlock(genChannels//4, genChannels//4, pixelNorm, leakyRelu, deConv=False),
                        A.GenBlock(genChannels//4, genChannels//8, pixelNorm, leakyRelu, deConv=True),
						A.GenBlock(genChannels//8, genChannels//8, pixelNorm, leakyRelu, deConv=False),
                        A.GenBlock(genChannels//8, genChannels//16, pixelNorm, leakyRelu, deConv=True),
						A.GenBlock(genChannels//16, genChannels//16, pixelNorm, leakyRelu, deConv=False),
                ])
                self.endblock = nn.Conv2d(genChannels//16, 2, kernel_size=1)

        def forward(self, x):
                #print(x.shape)
                assert len(x.shape)==2 and x.shape[1]==self.latentDim
                x = x.reshape(-1,self.latentDim,1,1)
                if self.normalize_latents:
                        mean = torch.mean(x*x, 1, keepdim=True)
                        dom = torch.rsqrt(mean + self.eps)
                        x = x * dom
                x = self.initblock(x)
                for block in self.blocks:
                        x = block(x)
                x = self.endblock(x)
                return x

class AudioEncoder(nn.Module):
        """docstring for AudioEncoder"""
        def __init__(self, latentDim=16, encChannels=256, specRatio=(3,8), specShape=(192,512), leakyRelu=True):
                super(AudioEncoder, self).__init__()
                self.latentDim = latentDim
                self.encChannels = encChannels
                self.specRatio = specRatio
                self.specShape = specShape
                self.leakyRelu = leakyRelu

                self.initblock = nn.Conv2d(2, encChannels//16, kernel_size=1)
                self.blocks = nn.ModuleList([
                        A.EncDiscBlock(encChannels//16, encChannels//8, leakyRelu, 'AvgPool2d'),
                        A.EncDiscBlock(encChannels//8, encChannels//8, leakyRelu, False),
                        A.EncDiscBlock(encChannels//8, encChannels//4, leakyRelu, 'AvgPool2d'),
                        A.EncDiscBlock(encChannels//4, encChannels//4, leakyRelu, False),
                        A.EncDiscBlock(encChannels//4, encChannels//2, leakyRelu, 'AvgPool2d'),
                        A.EncDiscBlock(encChannels//2, encChannels//2, leakyRelu, False),
                        A.EncDiscBlock(encChannels//2, encChannels, leakyRelu, 'AvgPool2d'),
                        A.EncDiscBlock(encChannels, encChannels, leakyRelu, False),
                        A.EncDiscBlock(encChannels, encChannels, leakyRelu, 'AvgPool2d'),
                        A.EncDiscBlock(encChannels, encChannels, leakyRelu, False),
                        A.EncDiscBlock(encChannels, encChannels, leakyRelu, 'AvgPool2d'),
                ])
                self.endblock = A.EncDiscEndBlock(latentDim, encChannels, specRatio, leakyRelu)

        def forward(self, x):
                x = self.initblock(x)
                for block in self.blocks:
                        x = block(x)
                x = self.endblock(x)
                return x

class AudioDiscriminator(nn.Module):
        """docstring for AudioDiscriminator"""
        def __init__(self, encChannels=256, specRatio=(3,8), specShape=(192,512), leakyRelu=True):
                super(AudioDiscriminator, self).__init__()
                self.encChannels = encChannels
                self.specRatio = specRatio
                self.specShape = specShape
                self.leakyRelu = leakyRelu

                self.initblock = nn.Conv2d(2, encChannels//16, kernel_size=1)
                self.blocks = nn.ModuleList([
                        A.EncDiscBlock(encChannels//16, encChannels//8, leakyRelu, 'AvgPool2d'),
                        A.EncDiscBlock(encChannels//8, encChannels//4, leakyRelu, 'AvgPool2d'),
                        A.EncDiscBlock(encChannels//4, encChannels//2, leakyRelu, 'AvgPool2d'),
                        A.EncDiscBlock(encChannels//2, encChannels, leakyRelu),
                        A.EncDiscBlock(encChannels, encChannels, leakyRelu),
                        A.EncDiscBlock(encChannels, encChannels, leakyRelu),
                ])
                self.endblock = A.EncDiscEndBlock(1, encChannels, specRatio, leakyRelu)

        def forward(self, x):
                x = self.initblock(x)
                for i in range(6):
                        x = self.blocks[i](x)
                x = self.endblock(x)
                return x

class ImageEncoder(object):
        """docstring for ImageEncoder"""
        def __init__(self, latentDim, modelName='resnet50'):
                super(ImageEncoder, self).__init__()
                self.latentDim = latentDim
                self.block = 'BottleNeck' if (int(modelName[6:]) > 34) else 'BasicBlock'
                self.resnet = getattr(torchvision.models, modelName)(pretrained=True)
                self.resnet.fc = nn.Linear(512*4 * (4 if self.block == 'BottleNeck' else 1), self.latentDim * 2)
        def forward(self, x):
                return self.resnet(x)