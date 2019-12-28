import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as udata
import torch.optim as optim
from tensorboardX import SummaryWriter
from dataset import HyperDataset
from FMNet import FMNet
from train import train, test
import time

parser = argparse.ArgumentParser(description="SpectralSR")
parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--batchSize_per_gpu", type=int, default=64, help="batch size")
parser.add_argument("--milestones", type=int, default=20, help="how many epoch to reduce the lr")
parser.add_argument("--gamma", type=int, default=0.2, help="how much to reduce the lr each time")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--opt.bNum", type=int, default=3, help='path log files')
parser.add_argument("--opt.nblocks", type=int, default=2, help='path log files')
parser.add_argument("--gpus", type=str, default="3,4,5", help='path log files')
opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

outf = 'logs/FMNet_original_{}_{}_{}_{}_{}_{}'.format(opt.bNum, opt.nblocks, opt.lr, opt.milestones, opt.gamma, opt.batchSize_per_gpu)

print(outf)
print(opt.gpus)


def main():
    gpu_num = len(opt.gpus.split(','))
    device_ids = list(range(gpu_num))
    print("loading dataset ...")
    train_dataset = HyperDataset(mode='train')
    test_dataset = HyperDataset(mode='test')
    batchSize = opt.batchSize_per_gpu * gpu_num
    train_loader = udata.DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=0)
    print('train dataset num : {}'.format(len(train_dataset)))
    criterion = nn.L1Loss()

    epoch = 0
    net = FMNet(bNum=opt.bNum, nblocks=opt.nblocks, input_features=31, num_features=64, out_features=31)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=1e-6, betas=(0.9, 0.999))
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.milestones, opt.gamma)
    load = False
    if load:
        checkpoint_file_name = 'checkpoint.pth'
        checkpoint = torch.load(os.path.join(outf, checkpoint_file_name), map_location=torch.device('cuda:0'))
        epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Successfully load checkpoint {} ... '.format(os.path.join(outf, checkpoint_file_name)))

    model = nn.DataParallel(net, device_ids=device_ids, output_device=device_ids[0])
    model.cuda()
    criterion.cuda()
    
    writer = SummaryWriter(outf)
    while epoch < opt.epochs:
        start = time.time()
        print("epoch {} learning rate {}".format(epoch, optimizer.param_groups[0]['lr']))
        
        train(model, criterion, optimizer, train_loader, epoch, writer)
        lr_scheduler.step()

        test(model, test_dataset, epoch, writer)

        if (epoch+1) % 20 == 0:
            torch.save({
                'epoch' : epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(outf, 'checkpoint_{}.pth'.format(epoch)))

        end = time.time()
        print('epoch {} cost {} hour '.format(epoch, str((end - start)/(60*60))))
        epoch += 1
    torch.save(model.module.state_dict(), os.path.join(outf, 'model.pth'))

if __name__ == "__main__":
    main()
