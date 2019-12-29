import torch
from utilities import batch_PSNR


def train(model, criterion, optimizer, train_loader, epoch, writer):
    train_times_per_epoch = len(train_loader)
    model.train()
    for i, data in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()
        real_hyper, _, real_rgb = data
        real_hyper, real_rgb = real_hyper.cuda(), real_rgb.cuda()
        fake_hyper,_ = model.forward(real_rgb)
        loss = criterion(fake_hyper, real_hyper)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            psnr = batch_PSNR(real_hyper, fake_hyper.detach())
            print("[epoch {}][{}/{}] psnr: {}".format(epoch, i, len(train_loader), psnr.item()))
            writer.add_scalar('train_psnr', psnr.item(), train_times_per_epoch*epoch + i)


def test(model, test_dataset, epoch, writer):
    test_image_num = len(test_dataset)
    model.eval()
    psnr_sum = 0
    for i, data in enumerate(test_dataset):
        real_hyper, _, real_rgb = data
        real_hyper, real_rgb = real_hyper.cuda(), real_rgb.cuda()
        fake_hyper,_ = model.forward(real_rgb)

        psnr = batch_PSNR(real_hyper, fake_hyper)
        print('test img [{}/{}], psnr {}'.format(i, test_image_num, psnr.item()))
        psnr_sum += psnr.item()
    
    print('total {} test images, avg psnr {}'.format(test_image_num, psnr_sum/test_image_num))
    writer.add_scalar('test_psnr', psnr_sum/test_image_num, epoch)