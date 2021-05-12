#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 08:43:31 2021

@author: apramanik
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias)




def upconv2x2(in_channels, out_channels, kernel, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='trilinear', scale_factor=2),
            conv3x3(in_channels, out_channels))



def create_fft(img):
    re,im=torch.split(img,[1,1],dim=1)
    re=re.squeeze(1)
    im=im.squeeze(1)
    img=torch.cat((re.unsqueeze(-1),im.unsqueeze(-1)),-1)
    x=torch.fft(img,2)
    re,im=torch.split(x,[1,1],dim=-1)
    re=re.squeeze(-1).unsqueeze(1)
    im=im.squeeze(-1).unsqueeze(1)
    ksp = torch.cat((re,im),1)
    return ksp


    
# create generator    
class generator(nn.Module):
    def __init__(self, out_channels=2, in_channels=10, start_filts=36, depth=4,
                 up_mode='transpose'):
        super(generator, self).__init__()
        
        
        self.conv_start = conv3x3(in_channels, start_filts)
        self.upconv1 = upconv2x2(int(start_filts), int(start_filts*2),[2,2])
        self.conv1 = conv3x3(int(start_filts*2), int(start_filts*4))
        self.upconv2 = upconv2x2(int(start_filts*4), int(start_filts*2),[2,2])
        self.conv2 = conv3x3(int(start_filts*2), int(start_filts))
        self.conv_final = conv3x3(start_filts, out_channels)
    
    def forward(self, x):
        x=self.conv_start(x)
        x=F.leaky_relu(self.upconv1(x),0.2)
        x=F.leaky_relu(self.conv1(x),0.2)
        x=F.leaky_relu(self.upconv2(x),0.2)
        x=F.leaky_relu(self.conv2(x),0.2)
        x=self.conv_final(x)
        return x

    

    
class dcblock(nn.Module):
    def __init__(self):
        super(dcblock, self).__init__()
    def forward(self, x, mask):
        ksp = create_fft(x)
        mask = mask.unsqueeze(1)
        mask = torch.cat((mask,mask),1)
        uksp = ksp*mask
        return uksp,mask



class usg(nn.Module):
    def __init__(self, lt_ch, lt_h, lt_w):
        super(usg, self).__init__()

        self.generator = generator(in_channels=lt_ch)                
        self.dcblock = dcblock()
        
    def forward(self, x, mask):
        fimg = self.generator(x)
        uksp,mmask = self.dcblock(fimg,mask)
        return uksp,fimg,mmask









if __name__ == "__main__":
    
    """
    lt_ch: channels of latent vector
    lt_h: height of latent vector
    lt_w: width of latent vector
    """
    lt_ch=2
    lt_h=64
    lt_w=40
    module = usg(lt_ch,lt_h,lt_w)
    x = torch.FloatTensor(np.random.random((4,lt_ch,lt_h,lt_w)))
    w = torch.FloatTensor(np.random.random((4,256,160)))
    y,z,f = module(x,w)
    print(module)
    print(y.size())
    print(z.size())
    print(f.size())

    