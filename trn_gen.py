#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 08:12:33 2021

@author: apramanik
"""




from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import os, torch, time
import torch.nn as nn
from scipy.io import savemat
from datetime import datetime
from skimage.measure import compare_psnr
from tqdm import tqdm


from data_prep import brain_data
from USG import usg
    


def save_checkpoint(states,  path, filename='model_best.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint_name = os.path.join(path,  filename)
    torch.save(states, checkpoint_name)

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


def train_unet(acc, sigma, learning_rate, lt_ch, lt_h, lt_w, ntrn, nSl, nSt, batch_Size, epochs, savemodNepoch):
    """
    Train on iowa96.
    """
    print ('*************************************************')
    start_time=time.time()
    saveDir='savedModels/'
    directory=saveDir+datetime.now().strftime("%d%b_%I%M%P_")+ str(ntrn)+ 'SUB_'+ str(nSl)+ 'Sl_' + str(nSt) + 'SSl_' + str(epochs)+'E_'+ str(sigma) + 'SN_' +str(acc)+'Fold_' +str(batch_Size)+'B_'+str(lt_ch)+'LTC_'+ str(lt_h) + 'LTH_' + str(lt_w) + 'LTW_' + str(learning_rate)+'lr_'+'USG'
    
    net = usg(lt_ch, lt_h, lt_w).cuda()
    loss_fn = nn.MSELoss().cuda()
    best_loss = np.float("inf")
    epochDCLoss,validLoss,ep=[],[],0
    zinit = np.zeros((batch_Size,lt_ch,lt_h,lt_w),dtype=np.float32)
    for i in range(batch_Size):
        zinit[i] = np.random.randn(lt_ch,lt_h,lt_w)
    z = torch.from_numpy(zinit.astype(np.float32)).cuda()
    z.requires_grad = True
    path = directory
    if not os.path.exists(path):
        os.makedirs(path)
        
    # create latent vector for every subject   
    for i in range(ntrn):
        ptnum = i+1
        zsub = np.zeros((nSl,lt_ch,lt_h,lt_w),dtype=np.float32)
        for i in range(nSl):
            zsub[i] = np.random.randn(lt_ch,lt_h,lt_w)
        np.save(directory+'/z_Patient'+str(ptnum)+'.npy',zsub)
    
    # define optimizer for generator and latent vector
    paramsG=list(net.generator.parameters()) + [z]
    Goptimizer = optim.Adam(paramsG, lr=learning_rate)
    # start training
    for epoch in tqdm(range(epochs)):
        net.train()
        ep=ep+1
        totalDCLoss=[]
        tr_dataset = brain_data('testing', ep, 1, ntrn, sigma, directory, lt_ch, lt_h, lt_w, nSl, nSt)
        tr_loader = DataLoader(tr_dataset, batch_size=batch_Size, shuffle=False, num_workers=0,pin_memory=True)
        for step, (uksp, gt_ksp, mask, rec_gt, dz, dptnum) in enumerate(tr_loader, 0):
            uksp, gt_ksp, mask, rec_gt, dz = uksp.cuda(), gt_ksp.cuda(), mask.cuda(), rec_gt.cuda(), dz.cuda()
            z.data = dz.data 
            recuksp,recimg,maskn  = net(z,mask)
            bmask=maskn.type(dtype=torch.BoolTensor).cuda()
            uksp = uksp.masked_select(bmask)
            recuksp = recuksp.masked_select(bmask)
            dcloss = loss_fn(recuksp,uksp)
            totalDCLoss.append(dcloss.detach().cpu().numpy())
            Goptimizer.zero_grad()
            dcloss.backward()
            Goptimizer.step()
            if step == 0:
                zpt = torch.clone(z)
            else:
                zpt = torch.cat((zpt,z),axis=0)
        zpt = zpt.detach().cpu().numpy().astype(np.float32)
        np.save(directory+'/z_Patient'+dptnum[0]+'.npy',zpt)
        epochDCLoss.append(np.mean(totalDCLoss))
        print("epoch: ", epoch, "DC loss: ", "%.5e" % epochDCLoss[epoch])

        
        # validation 
        with torch.no_grad():
            net.eval()
            vLoss = []
            psnr = np.zeros((nSl,))
            for step, (uksp, gt_ksp, mask, rec_gt, z, dptnum) in enumerate(tr_loader, 0):
                uksp, gt_ksp, mask, rec_gt, z = uksp.cuda(), gt_ksp.cuda(), mask.cuda(), rec_gt.cuda(), z.cuda()
                _,rec,_  = net(z,mask)
                recloss = loss_fn(rec, rec_gt)
                loss = recloss
                vLoss.append(loss.detach().cpu().numpy())
                rec = rec.detach().cpu().numpy()
                rec_gt = rec_gt.detach().cpu().numpy()
                rec=create_abs_np(rec)
                rec_gt=create_abs_np(rec_gt)
                psnr[step] = compare_psnr(rec, rec_gt)
            validLoss.append(np.mean(vLoss))
       

        print("epoch: ", epoch, "val loss: ", "%.5e" % validLoss[epoch])
        print("Mean Validation PSNR:", "%1.4f" % np.mean(psnr))
        
        # save models
        if np.remainder(ep,savemodNepoch)==0:
            save_checkpoint(
                {
            'epoch': ep,
            'state_dict': net.state_dict(),
            'best_loss': best_loss,
            'Goptimizer': Goptimizer.state_dict()
        },
        path=directory,
        filename='model-{}.pth.tar'.format(ep)
                )
    
        if validLoss[epoch] < best_loss:    
            best_loss = validLoss[epoch]
            save_checkpoint(
        {
            'epoch': ep,
            'state_dict': net.state_dict(),
            'best_loss': best_loss,
            'Goptimizer': Goptimizer.state_dict()
        },
        path=directory,
        )
    
    # save the loss related plots
    savemat(directory+'/epochDCLoss.mat',mdict={'epochs':epochDCLoss},appendmat=True)
    savemat(directory+'/validLoss.mat',mdict={'epochs':validLoss},appendmat=True)
    end_time = time.time()    
    print ('Training completed in minutes ', ((end_time - start_time) / 60))
    print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
    print ('*************************************************')
    for i in range(len(paramsG)):
        paramsG[i]=paramsG[i].detach().cpu().numpy()
    
    return epochDCLoss,validLoss,best_loss,paramsG




if __name__ == "__main__":
    
    """
    lt_ch: channels of latent vector
    lt_h: height of latent vector
    lt_w: width of latent vector
    ntrn: number of training subjects
    nSl: number of slices per subject
    nSt: slice number of each subject from where counting of slices start
    batch_Size: batch size
    sigma: noise variance
    acc: acceleration factor
    savemodNepoch: Save model after every N epochs
    epochDCLoss: data consistency loss every epoch
    validLoss: validation loss every epoch
    best_loss: least validation loss
    Gweights: Weights of generator
    """
    learning_rate=1e-3
    lt_ch=2 
    lt_h=64
    lt_w=40
    ntrn=2
    nSl=1
    nSt=100
    batch_Size=1
    sigma = 0.0
    acc=4
    epochs=10000
    savemodNepoch=5000
    epochDCLoss,validLoss,best_loss,Gweights=train_unet(acc, sigma, learning_rate, lt_ch, lt_h, lt_w, ntrn, nSl, nSt, batch_Size, epochs, savemodNepoch)
        
