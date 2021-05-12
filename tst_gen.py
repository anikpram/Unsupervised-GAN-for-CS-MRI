#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 08:29:03 2021

@author: apramanik
"""



from torch.utils.data import DataLoader
import numpy as np
import os, torch
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr, compare_ssim
from data_prep import brain_data
from USG import usg






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

#%%
"""
nSub: number of testing subjects
nSl: number of slices per subject
nSt: slice number of each subject from where counting of slices start
dispind: display index of the slice
sigma: noise variance
subDirectory: directory for saving model
lt_ch: channels of latent vector
lt_h: height of latent vector
lt_w: width of latent vector
"""
nSub=2
nSl=1
nSt=100
nImg=nSl*nSub
dispind=0
sigma = 0.0

#%%
############################## USG 2-fold #########################
subDirectory = '11May_1103am_2SUB_1Sl_100SSl_10000E_0.0SN_2Fold_1B_2LTC_64LTH_40LTW_0.001lr_USG'

################################# USG 4-fold ###############################################
#subDirectory = '11May_1214pm_2SUB_1Sl_100SSl_10000E_0.0SN_4Fold_1B_2LTC_64LTH_40LTW_0.001lr_USG'


print('The subdir for IDSLR is', subDirectory)


#%%
cwd=os.getcwd()
PATH= cwd+'/savedModels/'+subDirectory #complete path
directory = 'savedModels/'+subDirectory
lt_ch=2
lt_h=64
lt_w=40

#%%
# network
net = usg(lt_ch,lt_h,lt_w).cuda()
net.load_state_dict(torch.load(os.path.join(PATH, "model_best.pth.tar"))['state_dict'])
#net.load_state_dict(torch.load(os.path.join(PATH, "model-5000.pth.tar"))['state_dict'])
# list the parameters of network
params=list(net.parameters())
for i in range(len(params)):
        params[i]=params[i].detach().cpu().numpy()
weights_recon = params[:]
del params
normAtb=np.zeros((nImg,256,160),dtype=np.float32)
normOrg=np.zeros((nImg,256,160),dtype=np.float32)
normRec=np.zeros((nImg,256,160),dtype=np.float32)


psnr_rec = np.zeros((nImg,))
psnr_atb = np.zeros((nImg,))
ssim_rec = np.zeros((nImg,))
ssim_atb = np.zeros((nImg,))

#test the network
net.eval()
for tstsub in range(nSub):
    tst_dataset = brain_data('testing', tstsub+1, 1, 1, sigma, directory, lt_ch, lt_h, lt_w, nSl, nSt)
    tst_loader = DataLoader(tst_dataset, batch_size=1, shuffle=False, num_workers=0)
    for step, (uksp, gt_ksp, mask, rec_gt, z, ptnum) in enumerate(tst_loader, 0):
        uksp, gt_ksp, mask, rec_gt, z = uksp.cuda(), gt_ksp.cuda(), mask.cuda(), rec_gt.cuda(), z.cuda()
        _,rec,_  = net(z,mask)
        atb = create_abs_torch(create_ifft(uksp))
        rec = rec.detach().cpu().numpy().astype(np.float32)
        rec_gt = rec_gt.detach().cpu().numpy().astype(np.float32)
        atb = atb.detach().cpu().numpy()
        rec=create_abs_np(rec)
        rec_gt=create_abs_np(rec_gt)
        normAtb[(tstsub*nSl)+step]=atb
        normOrg[(tstsub*nSl)+step]=rec_gt
        normRec[(tstsub*nSl)+step]=rec
        psnr_rec[(tstsub*nSl)+step] = compare_psnr(rec, rec_gt)
        psnr_atb[(tstsub*nSl)+step] = compare_psnr(atb, rec_gt)
        ssim_rec[(tstsub*nSl)+step] = compare_ssim(rec[0], rec_gt[0])
        ssim_atb[(tstsub*nSl)+step] = compare_ssim(atb[0], rec_gt[0])
    mean_sub_fsnr = np.mean(psnr_rec[(tstsub*nSl):(tstsub+1)*nSl])
    mean_sub_isnr = np.mean(psnr_atb[(tstsub*nSl):(tstsub+1)*nSl])
    print("Initial PSNR: {0:.4f}".format(mean_sub_isnr)+" Final PSNR: {0:.4f}".format(mean_sub_fsnr))
        
        
normError=np.abs(normOrg-normRec).astype(np.float32)

    
print("Mean Initial PSNR: {0:.5f}".format(np.mean(psnr_atb)))
print("Mean Final PSNR: {0:.5f}".format(np.mean(psnr_rec)))
print("Mean Initial SSIM: {0:.5f}".format(np.mean(ssim_atb)))
print("Mean Final SSIM: {0:.5f}".format(np.mean(ssim_rec)))
 

#%% Display the output images
plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, interpolation='bilinear')
plt.clf()
plt.subplot(141)
st=0
end=256
plot(np.abs(normOrg[dispind,st:end,:]))
plt.axis('off')
plt.title('Fully Sampled')
plt.subplot(142)
plot(np.abs(normAtb[dispind,st:end,:]))
plt.title('Undersampled \n SNR='+str(psnr_atb[dispind].round(2))+' dB' )
plt.axis('off')
plt.subplot(143)
plot(np.abs(normRec[dispind,st:end,:]))
plt.title('USG, SNR='+ str(psnr_rec[dispind].round(2)) +' dB')
plt.axis('off')
plt.subplot(144)
plot(np.abs(normError[dispind,st:end,:]))
plt.title('Reconstruction Error')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=.01)
plt.show()



























