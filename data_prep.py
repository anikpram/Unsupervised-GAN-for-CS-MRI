#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 08:37:01 2021

@author: apramanik
"""



import numpy as np
import time
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.io import loadmat




#%% This provide functionality similar to matlab's tic() and toc()
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
    
#%%
def normalize_img(img):
    img = img.copy().astype(np.float32)
    img -= np.mean(img)
    img /= np.std(img)
    return img



def preprocess(kspr,nSl, nSt):
    ksp=kspr[:,:,:,0] + 1j*kspr[:,:,:,1]
    ksp=ksp.astype(np.complex64)
    del kspr
    img=np.zeros_like(ksp)
    for i in range(ksp.shape[0]):
        img[i] = np.fft.ifft2(ksp[i])
    img = np.swapaxes(img,0,-1)
    img = np.swapaxes(img,0,1)
    img = img[nSt:nSt+nSl,:,5:5+160] 
    ksp = np.zeros_like(img)
    for i in range(img.shape[0]):
        ksp[i] = np.fft.fft2(img[i]) 
    del img
    return ksp


def create_ifft_np(ksp):
    re,im=np.split(ksp,2,1)
    ksp=re+(1j*im)
    ksp = np.squeeze(ksp,1)
    img = np.zeros_like(ksp)
    for i in range(ksp.shape[0]):
        img[i] = np.fft.ifft2(ksp[i])
    img=c2romodl(img)
    return img



def create_ifft(ksp):
    re,im=torch.split(ksp,[1,1],dim=1)
    re=re.squeeze(1)
    im=im.squeeze(1)
    ksp=torch.cat((re.unsqueeze(-1),im.unsqueeze(-1)),-1)
    x=torch.ifft(ksp,2)
    re,im=torch.split(x,[1,1],dim=-1)
    re=re.squeeze(-1).unsqueeze(1)
    im=im.squeeze(-1).unsqueeze(1)
    img = torch.cat((re,im),1)
    return img

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
    


def create_abs_torch(img):
    re,im = torch.split(img,[1,1],dim=1)
    re = torch.abs(re.squeeze(1))
    im = torch.abs(im.squeeze(1))
    img = torch.sqrt((re*re) + (im*im))
    return img

def create_abs_np(img):
    re,im=np.split(img,2,1)
    img = re+1j*im
    img = np.abs(img)
    img = np.squeeze(img,1)
    return img
    


def generate_undersampled(org,sigma):
    nSl = org.shape[0]
    std = np.std(org)
    org = org/std
    if sigma==0.0:
        kspn = org[:]
    else:
        noise = np.random.randn(nSl,256,160) + 1j*np.random.randn(nSl,256,160)
        noise = sigma * noise[:]
        kspn = org[:] + noise[:]
    ksp=c2romodl(org)
    msk = loadmat('mask4f_256by160_100m.mat')['a']
    rn = np.random.randint(100)
    msk = msk[rn]
    msk = np.abs(np.fft.ifftshift(msk))
    msk=msk.astype(np.float32)
    msk=np.expand_dims(msk,axis=0)
    msk = np.tile(msk,[org.shape[0],1,1])
    uksp=c2romodl(kspn*msk)
    return ksp,uksp,msk


#%%

def swapaxes(ksp):
    nSlice,nrow,ncol,ncoil=ksp.shape
    kspn=np.zeros((nSlice,ncoil,nrow,ncol),dtype=ksp.dtype)
    for i in range(ncoil):
        kspn[:,i,:,:]=ksp[:,:,:,i]
    return kspn



        


def r2comodl(inp):
    """  input img: row x col x 2 in float32
    output image: row  x col in complex64
    """
    if inp.dtype=='float32':
        dtype=np.complex64
    else:
        dtype=np.complex128
    out=np.zeros((inp.shape[0],inp.shape[1],inp.shape[2],int(inp.shape[3]/2)),dtype=dtype)
    re,im=np.split(inp,2,axis=-1)
    out=re+(1j*im)
    return out


def r2comodl_tensor(inp):
    """  input img: row x col x 2 in float32
    output image: row  x col in complex64
    """
    inp=inp.numpy
    if inp.dtype=='float32':
        dtype=np.complex64
    else:
        dtype=np.complex128
    out=np.zeros((inp.shape[0],inp.shape[1],inp.shape[2],int(inp.shape[3]/2)),dtype=dtype)
    re,im=np.split(inp,2,axis=-1)
    out=re+(1j*im)
    return out

def c2romodl(inp):
    """  input img: row x col in complex64
    output image: row  x col x2 in float32
    """
    if inp.dtype=='complex64':
        dtype=np.float32
    else:
        dtype=np.float64
    out=np.zeros((inp.shape[0],2,inp.shape[1],inp.shape[2]),dtype=dtype)
    out[:,0,:,:]=np.real(inp)
    out[:,1,:,:]=np.imag(inp)
    return out





#%%paths
IMG_DIR = "/Shared/lss_jcb/aniket/Calgary Single Channel brain dataset/"


#%%dataset preparation
case_list = list(range(1,26))
tr_case = case_list[:]
val_case = case_list[:10]
test_case = case_list[:10]


class brain_data(Dataset):
    """
    dset: dataset type
    epoch: epoch number
    chunk_size: number of subjects per epoch
    ntrn: number of training subjects
    sigma: noise variance
    zdir: directory from where latent vector is to be loaded
    lt_ch: channels of latent vector
    lt_h: height of latent vector
    lt_w: width of latent vector
    ntrn: number of training subjects
    nSl: number of slices per subject
    """

    def __init__(self, dset, epoch, chunk_size, ntrn, sigma, zdir, lt_ch, lt_h, lt_w, nSl, nSt):        
        if dset == 'training':
            st=(epoch-1)*chunk_size
            st=st%ntrn
            if (st+chunk_size)>ntrn:
                b=(st+chunk_size)%ntrn
                self.subj = tr_case[st:ntrn]+tr_case[:b]
                self.datafolder = 'Train'
            else:
                self.subj = tr_case[st:st+chunk_size]
                self.datafolder = 'Train'
            
        elif dset == 'validation':
            st=(epoch-1)*chunk_size
            st=st%ntrn
            if (st+chunk_size)>ntrn:
                b=(st+chunk_size)%ntrn
                self.subj = val_case[st:ntrn]+val_case[:b]
                self.datafolder = 'Val'
            else:
                self.subj = val_case[st:st+chunk_size]
                self.datafolder = 'Val'
        else:
            st=(epoch-1)*chunk_size
            st=st%ntrn
            if (st+chunk_size)>ntrn:
                b=(st+chunk_size)%ntrn
                self.subj = test_case[st:ntrn]+test_case[:b]
                self.datafolder = 'Val'
            else:
                self.subj = test_case[st:st+chunk_size]
                self.datafolder = 'Val'

        
        atb_ksp = np.zeros((1,2,256,160),dtype=np.float32)
        org_ksp = np.zeros((1,2,256,160),dtype=np.float32)
        us_mask = np.zeros((1,256,160),dtype=np.float32)
        org_img = np.zeros((1,2,256,160),dtype=np.float32)
        lt_vec = np.zeros((1,lt_ch,lt_h,lt_w),dtype=np.float32)
        
        for k in range(len(self.subj)):
            j = self.subj[k]
            print(j)
            ptnum = str(j)
        
            img_dir = IMG_DIR + self.datafolder + '/Patient'+ptnum+'_1ch.npy'
            kspr = np.load(img_dir)
            ksp=preprocess(kspr, nSl, nSt)
            dummy_ksp, dummy_atb_ksp, dummy_mask = generate_undersampled(ksp, sigma)
            dummy_img=create_ifft_np(dummy_ksp)
            z=np.load(zdir+'/z_Patient'+ptnum+'.npy')
            
            
            atb_ksp = np.concatenate((atb_ksp,dummy_atb_ksp),axis=0)
            org_ksp = np.concatenate((org_ksp,dummy_ksp),axis=0)
            us_mask = np.concatenate((us_mask, dummy_mask),axis=0)
            org_img = np.concatenate((org_img,dummy_img),axis=0)
            lt_vec = np.concatenate((lt_vec,z),axis=0)
            
            atb_ksp = atb_ksp[1:,:,:,:]
            org_ksp = org_ksp[1:,:,:,:]
            us_mask = us_mask[1:,:,:]
            org_img = org_img[1:,:,:,:]
            lt_vec = lt_vec[1:,:,:,:]
            
            self.atb = atb_ksp[:]
            self.orgk = org_ksp[:]
            self.mask = us_mask[:]
            self.orgi = org_img[:]
            self.lt_vec = lt_vec[:]
            self.ptnum = ptnum
            self.len = atb_ksp.shape[0]
            
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        
        dorgk = self.orgk[i]
        datb = self.atb[i]
        dmask = self.mask[i]
        dorgi = self.orgi[i]
        dlvec = self.lt_vec[i]
        dptnum = self.ptnum
        
        dksp = torch.from_numpy(dorgk.astype(np.float32))
        duksp = torch.from_numpy(datb.astype(np.float32))
        dusmask = torch.from_numpy(dmask.astype(np.float32))
        dimg = torch.from_numpy(dorgi.astype(np.float32))
        dz = torch.from_numpy(dlvec.astype(np.float32))

        return duksp,dksp,dusmask,dimg,dz,dptnum
    
    
if __name__ == "__main__":
    zdir = 'savedModels/11May_1103am_2SUB_1Sl_100SSl_10000E_0.0SN_2Fold_1B_2LTC_64LTH_40LTW_0.001lr_USG' 
    dataset = brain_data('testing', 17, 1, 2, 0.0, zdir, 2, 64, 40, 1, 100)
    loader = DataLoader(dataset, shuffle=False, batch_size=2)
    count=0
    dispind = 0
    for step, (atb_ksp, gt_ksp, mask, gt_img, z, dptnum) in enumerate(loader):
        print(atb_ksp.numpy().max())
        print(atb_ksp.numpy().min())
        atb = create_abs_torch(create_ifft(atb_ksp))
        gt = create_abs_torch(create_ifft(gt_ksp))
        gt_img = create_abs_torch(gt_img)
        fig, axes = plt.subplots(1,3)
        pos = axes[0].imshow(gt[dispind,])
        pos = axes[1].imshow(gt_img[dispind,])
        pos = axes[2].imshow(atb[dispind,])
        plt.show()
        break
        