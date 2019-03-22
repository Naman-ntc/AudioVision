import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

ngroups=1

class Identity(nn.Module):
        """docstring for Identity"""
        def __init__(self):
                super(Identity, self).__init__()
        def forward(self, x):
                return x                

class PixelNorm(nn.Module):
        """docstring for PixelNorm"""
        def __init__(self):
                super(PixelNorm, self).__init__()
                self.eps = 1e-8

        def forward(self, x):
                mean = torch.mean(x*x, 1, keepdim=True)
                dom = torch.rsqrt(mean + self.eps)
                x = x * dom
                return x

class GenInitBlock(nn.Module):
        """docstring for GenInitBlock"""
        def __init__(self, latentDim=256, genChannels=256, specRatio=(3,8), pixelNorm=True, leakyRelu=True):
                super(GenInitBlock, self).__init__()
                self.latentDim = latentDim
                self.genChannels = genChannels
                self.specRatio = specRatio
                self.pixelNorm = pixelNorm
                self.leakyRelu = leakyRelu

                self.layer1 = [
                                nn.ConvTranspose2d(self.latentDim, self.genChannels, self.specRatio, groups=ngroups),
                                nn.LeakyReLU(0.2) if leakyRelu else nn.ReLU(),
                        ] + [nn.GroupNorm(ngroups, self.genChannels)]
                self.layer2 = [
                                nn.Conv2d(self.genChannels, self.genChannels, kernel_size=3, padding=1, groups=ngroups),
                                nn.LeakyReLU(0.2) if leakyRelu else nn.ReLU(),
                        ] + [nn.GroupNorm(ngroups, self.genChannels)]
                self.layer = nn.Sequential(*(self.layer1+self.layer2))
                
        def forward(self, x):
                return self.layer(x)

class GenBlock(nn.Module):
        """docstring for GenBlock"""
        def __init__(self, genChannels1=256, genChannels2=256, pixelNorm=True, leakyRelu=True, deConv=True):
                super(GenBlock, self).__init__()
                self.genChannels1 = genChannels1
                self.genChannels2 = genChannels2
                self.pixelNorm = pixelNorm
                self.leakyRelu = leakyRelu
                self.deConv = deConv
                self.channels = nn.Conv2d(self.genChannels1, self.genChannels2, kernel_size=1, groups=ngroups) 
                self.layer1 = [
                                nn.Conv2d(self.genChannels2, self.genChannels2, kernel_size=3, padding=1, groups=ngroups),
                                nn.LeakyReLU(0.2) if leakyRelu else nn.ReLU(),
                        ] + [nn.GroupNorm(ngroups, self.genChannels2)]
                self.layer2 = [
                                nn.Conv2d(self.genChannels2, self.genChannels2, kernel_size=3, padding=1, groups=ngroups),
                                nn.LeakyReLU(0.2) if leakyRelu else nn.ReLU(),
                        ] + [nn.GroupNorm(ngroups, self.genChannels2)]
                self.layer1 = nn.Sequential(*self.layer1)
                self.layer2 = nn.Sequential(*self.layer2)
                #self.layer = nn.Sequential(*(self.layer1 + self.layer2))
                #self.upsample = nn.ConvTranspose2d(self.genChannels2, self.genChannels2, kernel_size=4, padding=1)#, output_padding=1)
                self.upsample = nn.ConvTranspose2d(self.genChannels2, self.genChannels2, kernel_size=4, padding=1, stride=2) if self.deConv else Identity()
                #, output_padding=1)
        def forward(self, x):
                x = self.channels(x)
                x = self.layer1(x) + x
                x = self.layer2(x) + x
                x = self.upsample(x)
                return x

class EncDiscBlock(nn.Module):
        """docstring for EncDiscBlock"""
        def __init__(self, encChannels1=256, encChannels2=256, leakyRelu=True, Pooling= 'MaxPool2d'):
                super(EncDiscBlock, self).__init__()
                self.encChannels1 = encChannels1
                self.encChannels2 = encChannels2
                self.leakyRelu = leakyRelu
                self.Pooling = Pooling

                self.pooling = getattr(nn, self.Pooling)(2,2) if self.Pooling else Identity()
                self.channels = nn.Conv2d(self.encChannels1, self.encChannels2, kernel_size=1, groups=ngroups) 
                self.layer1 = [
                                nn.GroupNorm(ngroups, self.encChannels2),
                                nn.Conv2d(self.encChannels2, self.encChannels2, kernel_size=3, padding=1, groups=ngroups),
                                nn.LeakyReLU(0.2) if leakyRelu else nn.ReLU(),
                        ]
                self.layer2 = [
                                nn.GroupNorm(ngroups, self.encChannels2),
                                nn.Conv2d(self.encChannels2, self.encChannels2, kernel_size=3, padding=1, groups=ngroups),
                                nn.LeakyReLU(0.2) if leakyRelu else nn.ReLU(),
                        ]
                self.layer1 = nn.Sequential(*self.layer1)
                self.layer2 = nn.Sequential(*self.layer2)


        def forward(self, x):
                x = self.channels(self.pooling(x))
                x = self.layer1(x) + x
                x = self.layer2(x) + x
                return x

class EncDiscEndBlock(nn.Module):
        """docstring for EncDiscEndBlock"""
        def __init__(self, latentDim=256, encChannels=256, specRatio=(3,8), leakyRelu=True, Pooling='MaxPool2d'):
                super(EncDiscEndBlock, self).__init__()
                self.latentDim = latentDim
                self.encChannels = encChannels
                self.specRatio = specRatio
                self.leakyRelu = leakyRelu
                self.Pooling = Pooling

                self.layer1 = [
                                nn.GroupNorm(ngroups, self.encChannels),
                                nn.Conv2d(self.encChannels, self.encChannels, kernel_size=3, padding=1, groups=ngroups),
                                nn.LeakyReLU(0.2) if leakyRelu else nn.ReLU(),
                        ]
                self.layer2 = [
                                nn.GroupNorm(ngroups, self.encChannels),
                                nn.Conv2d(self.encChannels, self.latentDim, kernel_size=specRatio, groups=ngroups),
                        ]
                self.layer = nn.Sequential(*(self.layer1 + self.layer2))

        def forward(self, x):
                return self.layer(x)
