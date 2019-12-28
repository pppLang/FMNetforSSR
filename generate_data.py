import os
import h5py
import numpy as np
import cv2
from utilities import Im2Patch
from scipy.interpolate import interp1d
from shutil import copyfile

def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)

def process_data(index, key, patch_size, stride, h5f, hyper_name, rgb_name, mode):
    mat =  h5py.File(hyper_name,'r')
    hyper = np.float32(np.array(mat['rad']))
    hyper = np.transpose(hyper, [0,2,1])
    hyper = normalize(hyper, max_val=4095., min_val=0.)
    mat.close()
    # load rgb image
    rgb =  cv2.imread(rgb_name)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = np.transpose(rgb, [2,0,1])
    rgb = normalize(np.float32(rgb), max_val=255., min_val=0.)
    # rgb插值
    real_rgb = rgb # 3x64x64
    print(real_rgb.shape)
    zeros = np.ones([28, real_rgb.shape[1], real_rgb.shape[2]], dtype=np.float32)
    real_rgb = np.append(real_rgb, zeros, axis=0)
    x1 = np.linspace(0, 1, num=3)
    x2 = np.linspace(0, 1, num=31)
    for m in range(real_rgb.shape[1]):
        for j in range(real_rgb.shape[2]):
            f = interp1d(x1, real_rgb[0:3,m,j])
            real_rgb[:,m,j] = f(x2)

    if mode == 'train':
        # creat patches
        patches_hyper = Im2Patch(hyper, win=patch_size, stride=stride)
        patches_rgb = Im2Patch(rgb, win=patch_size, stride=stride)
        patches_real_rgb = Im2Patch(real_rgb, win=patch_size, stride=stride)
        # add data
        for j in range(patches_hyper.shape[3]):
            print("generate training sample {}".format(index))
            sub_hyper = patches_hyper[:,:,:,j]
            sub_rgb = patches_rgb[:,:,:,j]
            sub_real_rgb = patches_real_rgb[:,:,:,j]
            data = np.concatenate((sub_hyper, sub_rgb, sub_real_rgb), 0)
            h5f.create_dataset(str(index), data=data)
            index += 1
    elif mode == 'test':
        data = np.concatenate((hyper, rgb, real_rgb), 0)
        h5f.create_dataset(key, data=data)
        index += 1
    return index


path = '/data0/langzhiqiang/data'
train_names_file = open(os.path.join(path, 'train_names.txt'), mode='r')
test_names_file = open(os.path.join(path, 'test_names.txt'), mode='r')
train_num = 0
test_num = 0

h5f = h5py.File('/data0/langzhiqiang/data/train_realworld.h5', 'w')
index = 1
for line in train_names_file:
    line = line.split('\n')[0]
    hyper_rgb = line.split(' ')
    hyper_name, rgb_name = hyper_rgb[0], hyper_rgb[1]
    # rgb_name = rgb_name.replace('Train_RGB', 'Train_RGB_RealWorld').replace('clean.png', 'camera.jpg')  # generate Real World
    print(hyper_name, rgb_name)
    index = process_data(index, None, 64, 64, h5f, hyper_name, rgb_name, mode='train')
    train_num += 1
h5f.close()
train_patches = index
print('train patches num : {}'.format(train_patches))


h5f = h5py.File('/data0/langzhiqiang/data/test.h5', 'w')
index = 1
for line in test_names_file:
    line = line.split('\n')[0]
    hyper_rgb = line.split(' ')
    hyper_name, rgb_name = hyper_rgb[0], hyper_rgb[1]
    print(hyper_name, rgb_name)
    key = int(hyper_name.split('.')[0].split('_')[-1])
    print(key)
    index = process_data(index, key, None, None, h5f, hyper_name, rgb_name, mode='test')
    test_num += 1
h5f.close()
test_patches = index
print('test patches num : {}'.format(test_patches))

print('{} train file, {} train patches'.format(train_num, train_patches))
print('{} test file, {} test patch'.format(test_num, test_patches))