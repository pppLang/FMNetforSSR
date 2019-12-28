import random
import glob
import os
import cv2


path='/data0/langzhiqiang/data'

filenames_hyper = glob.glob(os.path.join(path,'Train_Spectral','*.mat'))
filenames_rgb = glob.glob(os.path.join(path,'Train_RGB','*.png'))
filenames_hyper.sort()
filenames_rgb.sort()

assert len(filenames_hyper) == len(filenames_rgb)
num = len(filenames_hyper)
all_ = list(range(num))
train_index = random.sample(all_, 200)  # sample 200 image pairs for training
test_index = [i for i in all_ if i not in train_index]
train_index.sort()
test_index.sort()

print('total {} image pairs, {} training, {} testing'.format(num, len(train_index), len(test_index)))

train_num = 0
test_num = 0
train_names_file = open(os.path.join(path, 'train_names.txt'), mode='x')
test_names_file = open(os.path.join(path, 'test_names.txt'), mode='x')

for i in range(num):
    if i in train_index:
        train_names_file.write(filenames_hyper[i] + ' ' + filenames_rgb[i] + '\n')
        train_num+=1
    elif i in test_index:
        test_names_file.write(filenames_hyper[i] + ' ' + filenames_rgb[i] + '\n')
        test_num+=1
    else:
        print('error !!!!!!!!!')
        assert False

print('train file num : {}'.format(train_num))
print('test file num : {}'.format(test_num))
train_names_file.close()
test_names_file.close()