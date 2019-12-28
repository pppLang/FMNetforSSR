import os
import torch
import numpy as np 


def batch_PSNR(im_true, im_fake, data_range=255):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    Ifake = im_fake.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C*H*W)
    psnr = 10. * torch.log((data_range**2)/err) / np.log(10.)
    return torch.mean(psnr)

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])