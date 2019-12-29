import os
import torch
import numpy as np
import scipy.io as scio
from tensorboardX import SummaryWriter
from FMNet import FMNet
import glob
from dataset import HyperDataset
from utilities import batch_PSNR
import time
import argparse

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

parser = argparse.ArgumentParser(description="SpectralSR")
parser.add_argument("--img_id", type=int, default="51", help='path log files')
opt = parser.parse_args()

def get_testfile_list():
    path = '/data0/langzhiqiang/data'
    test_names_file = open(os.path.join(path, 'test_names.txt'), mode='r')

    test_rgb_filename_list = []
    for line in test_names_file:
        line = line.split('/n')[0]
        hyper_rgb = line.split(' ')[0]
        test_rgb_filename_list.append(hyper_rgb)

    return test_rgb_filename_list


def main(model_path, need_weisout=False):

    if not os.path.exists(os.path.join(model_path, 'result')):
        os.mkdir(os.path.join(model_path, 'result'))
    if not os.path.exists(os.path.join(model_path, 'mask')):
        os.mkdir(os.path.join(model_path, 'mask'))

    print('load model path : {}'.format(model_path))
    model_name = 'model.pth'
    name = model_path.split('/')[-1].split('_')
    bNum = int(name[2])
    nblocks = int(name[3])
    model = FMNet(bNum=bNum, nblocks=nblocks, input_features=31, num_features=64, out_features=31)
    print(bNum, nblocks)
    
    model.load_state_dict(torch.load(os.path.join(model_path, model_name), map_location='cpu'))
    print('model MNet has load !')

    model.eval()
    model.cuda()
    
    testDataset = HyperDataset(mode='test')
    num = len(testDataset)
    print('test img num : {}'.format(num))

    test_rgb_filename_list = get_testfile_list()
    print('test_rgb_filename_list len : {}'.format(len(test_rgb_filename_list)))

    psnr_sum = 0
    all_time = 0
    for i in range(num):
        
        file_name = test_rgb_filename_list[i].split('/')[-1]
        key = int(file_name.split('.')[0].split('_')[-1])
        print(file_name, key)

        real_hyper, _, real_rgb = testDataset.get_data_by_key(str(key))
        real_hyper, real_rgb = torch.unsqueeze(real_hyper, 0), torch.unsqueeze(real_rgb, 0)
        real_hyper, real_rgb = real_hyper.cuda(), real_rgb.cuda()

        # print('test img [{}/{}], input rgb shape : {}, hyper shape : {}'.format(i+1, num, real_rgb.shape, real_hyper.shape))
        # forward
        with torch.no_grad():
            start = time.time()
            fake_hyper = model.forward(real_rgb)
            all_time += (time.time() - start)

        if isinstance(fake_hyper, tuple):
            fake_hyper, weis_out = fake_hyper
            if need_weisout:
                weis_out = weis_out[0,:,0,:,:].cpu().numpy()
                weis_out = np.squeeze(weis_out)
            else:
                weis_out = False
        else:
            weis_out = None

        psnr = batch_PSNR(real_hyper, fake_hyper).item()
        print('test img [{}/{}], fake hyper shape : {}, psnr : {}'.format(i+1, num, fake_hyper.shape, psnr))
        psnr_sum += psnr
        fake_hyper_mat = fake_hyper[0,:,:,:].cpu().numpy()
        if weis_out is None:
            scio.savemat(os.path.join(model_path, 'result', test_rgb_filename_list[i].split('/')[-1].split('.')[0]+'.mat'), {'rad': fake_hyper_mat})
            print('sucessfully save fake hyper !!!')
        else:
            scio.savemat(os.path.join(model_path, 'result', test_rgb_filename_list[i].split('/')[-1].split('.')[0]+'.mat'), {'rad': fake_hyper_mat, 'weis_out': weis_out})
            print('sucessfully save fake hyper and weis_out !!!')
        print()

    print('average psnr : {}'.format(psnr_sum/num))
    print('average test time : {}'.format(all_time/num))


if __name__ == "__main__":
    dir_list = ['logs/FMNet_original_3_2_0.0001_[20, 40, 60, 80]_0.5_64']
    for path in dir_list:
        print(os.path.join(path, 'model.pth'))
        if os.path.exists(os.path.join(path, 'model.pth')):
            print(path)
            main(path, need_weisout=True)