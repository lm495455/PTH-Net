import argparse
import os
import time
import shutil
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.ETH_Net import eth_net
import matplotlib
import random
from thop import profile

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from dataloader.dataset import train_data_loader, test_data_loader
import combine_test as test

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--bz', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, metavar='LR', dest='lr')
parser.add_argument('--gamma', '--scheduler-gamma', default=0.5, type=float)
parser.add_argument('--mil', '--milestones', default=[5, 10, 15, 20], type=list)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--dataset', type=str, default='DFEW',
                    choices=['DFEW', 'FERv39k', 'MAFW', 'CREMA-D', 'eNTERFACE05', 'RAVDESS', 'MELD', 'CASME2'])
parser.add_argument('--data_mode', type=str, default='rv', choices=['norm', 'rv'])
parser.add_argument('--label_type', type=str, default='single', choices=['single', 'compound'])
parser.add_argument('--num_class', type=int, default=7, choices=[5, 6, 7, 8, 11, 43])
parser.add_argument('--is_face', type=bool, default=True)
parser.add_argument('--mes', type=str, default='DFEW')
parser.add_argument('--data_set', type=int, default=2)
parser.add_argument('--gpu', type=str, default='3')
# eth_net 参数
parser.add_argument('--max_len', type=int, default=16)
parser.add_argument('--k', type=list, default=[1, 3, 5])
parser.add_argument('--thr_size', type=list, default=[3, 1, 3, 3])
parser.add_argument('--arch', type=tuple, default=(2, 2, 1, 1))
parser.add_argument('--n_in', type=int, default=1408)
parser.add_argument('--n_embd', type=int, default=512)
parser.add_argument('--downsample_type', type=str, default='max')
parser.add_argument('--scale_factor', type=int, default=2)
parser.add_argument('--with_ln', type=bool, default=True)
parser.add_argument('--mlp_dim', type=int, default=768)
parser.add_argument('--path_pdrop', type=int, default=0.1)
parser.add_argument('--use_pos', type=bool, default=False)

def Run(args, train_loader, val_loader, log_txt_path, log_curve_path, checkpoint_path, best_checkpoint_path,
        save=False):
    best_acc = float('-inf')
    best_F1 = float('-inf')
    recorder = RecorderMeter(args.epochs)
    print('The training set: set ' + str(args.data_set))
    with open(log_txt_path, 'a') as f:
        f.write('The training set: set ' + str(args.data_set) + '\n')

    # create model and load pre_trained parameters
    model = eth_net(args.n_in, args.n_embd, args.mlp_dim, args.max_len, args.arch, args.scale_factor, args.with_ln,
                    args.path_pdrop, args.downsample_type, args.thr_size, args.k, use_pos=args.use_pos,
                    num_classes=args.num_class)
    # input_4, input_8, input = torch.randn(1, 1408, 4), torch.randn(1, 1408, 8), torch.randn(1, 1408, 16)
    # mask_4, mask_8, mask = torch.ones(1, 1, 4), torch.ones(1, 1, 8), torch.ones(1, 1, 16)
    # flops, params = profile(model, inputs=(input_4, input_8, input, mask_4, mask_8, mask, True))
    # print('the flops is {}G,the params is {}M'.format(round(flops / (10 ** 9), 2), round(params / (10 ** 6), 2)))
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.mil, gamma=args.gamma)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    for epoch in range(args.start_epoch, args.epochs):
        if epoch == 0 and save:
            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                        'recorder': recorder}, checkpoint_path.replace('model', 'model_begin'))
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        print(inf)
        print('Current learning rate: ', current_learning_rate)

        # train for one epoch
        train_acc, train_los, train_F1 = train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path)

        # evaluate on validation set
        if epoch > args.epochs * -1:
            val_acc, val_los,  val_F1 = validate(val_loader, model, criterion, args, log_txt_path)

            # remember best acc and save checkpoint
            is_best = val_acc > best_acc
            if is_best:
                best_F1 = val_F1
            best_acc = max(val_acc, best_acc)
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'best_acc': best_acc,
                             'optimizer': optimizer.state_dict(),
                             'recorder': recorder}, is_best, checkpoint_path, best_checkpoint_path)

            # print and save log
            epoch_time = time.time() - start_time
            recorder.update(epoch, train_los, train_acc, val_los, val_acc)
            recorder.plot_curve(log_curve_path)

            print('The best accuracy: {:.3f}'.format(best_acc.item()))
            print('The best F1 Score: {:.3f}'.format(val_F1.item()))
            print('An epoch time: {:.1f}s'.format(epoch_time))
            with open(log_txt_path, 'a') as f:
                f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
                f.write('The best F1 Score: ' + str(val_F1.item()) + '\n')
                f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')
        scheduler.step()
    return best_acc, best_F1


def train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path):
    ls = AverageMeter('Loss', ':.4f')
    ls_ce = AverageMeter('loss_ce', ':.4f')
    ls_L = AverageMeter('loss_L', ':.4f')
    ls_M = AverageMeter('loss_M', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    top1_F1 = AverageMeter('F1 Score', ':6.4f')
    progress = ProgressMeter(len(train_loader),
                             [ls, ls_ce, ls_L, ls_M, top1, top1_F1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    a = 1
    for i, (input_H, input_M, input_L, target) in enumerate(train_loader):
        input_L, input_M, input_H, masks_L, masks_M, masks_H = input_L[0].cuda(), input_M[0].cuda(), input_H[0].cuda(), \
            input_L[1].cuda(), input_M[1].cuda(), input_H[1].cuda()
        target = target.cuda()
        pred_L, pred_M, pred = model(input_L, input_M, input_H, masks_L, masks_M, masks_H, True)
        loss_ce = criterion(pred, target)
        acc1, _ = accuracy(pred, target, topk=(1, 5))
        # 计算pred与target的F1 Score
        temp = torch.argmax(pred, dim=-1)
        F1_Score = f1_score(target.cpu(), temp.cpu().detach().numpy(), average='weighted')
        loss_L = a * nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(dim=1)(pred_L),
                                                         nn.Softmax(dim=1)(pred.detach()))
        loss_M = a * nn.KLDivLoss(reduction='batchmean')(nn.LogSoftmax(dim=1)(pred_M),
                                                         nn.Softmax(dim=1)(pred.detach()))

        loss = loss_ce + loss_L + loss_M

        ls.update(loss.item(), input_L.size(0))
        ls_L.update(loss_L.item(), input_L.size(0))
        ls_M.update(loss_M.item(), input_L.size(0))
        ls_ce.update(loss_ce.item(), input_L.size(0))
        top1.update(acc1[0], input_L.size(0))
        top1_F1.update(F1_Score, input_L.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i, log_txt_path)
    return top1.avg, ls.avg, top1_F1.avg


def validate(val_loader, model, criterion, args, log_txt_path):
    ls = AverageMeter('Loss', ':.4f')
    ls_ce = AverageMeter('loss_ce', ':.4f')
    ls_L = AverageMeter('loss_L', ':.4f')
    ls_M = AverageMeter('loss_M', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    top1_F1 = AverageMeter('F1 Score', ':6.4f')
    progress = ProgressMeter(len(val_loader),
                             [ls, ls_ce, ls_L, ls_M, top1,  top1_F1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input_H, input_M, input_L, target) in enumerate(val_loader):
            input_L, input_M, input_H, masks_L, masks_M, masks_H = input_L[0].cuda(), input_M[0].cuda(), input_H[
                0].cuda(), \
                input_L[1].cuda(), input_M[1].cuda(), input_H[1].cuda()
            target = target.cuda()
            # compute output
            pred = model(None, None, input_H, None, None, masks_H, False)
            loss_ce = criterion(pred, target)
            acc1, _ = accuracy(pred, target, topk=(1, 5))
            # 计算pred与target的F1 Score
            temp = torch.argmax(pred, dim=-1)
            F1_Score = f1_score(target.cpu(), temp.cpu().detach().numpy(), average='weighted')
            # pred_H = model(None, None, input_H, None, None, masks_H, False)
            # pred_M = model(None, input_M, None, None, masks_M, None, False)
            # pred_L = model(input_L, None, None, masks_L, None, None, False)
            # loss_ce = criterion(pred_H, target)
            # acc1_H, _ = accuracy(pred_H, target, topk=(1, 5))
            # acc1_M, _ = accuracy(pred_M, target, topk=(1, 5))
            # acc1_L, _ = accuracy(pred_L, target, topk=(1, 5))
            # # acc1 = (acc1_H + acc1_M + acc1_L) / 3
            # acc1 = max(acc1_H, acc1_L, acc1_M)

            # pred_L, pred_M, pred = model(input_L, input_M, input_H, masks_L, masks_M, masks_H, True)
            # loss_ce = criterion(pred, target)
            # acc1, _ = accuracy(pred, target, topk=(1, 5))

            ls.update(loss_ce.item(), input_L.size(0))
            ls_L.update(0, input_L.size(0))
            ls_M.update(0, input_L.size(0))
            ls_ce.update(loss_ce.item(), input_L.size(0))
            top1.update(acc1[0], input_L.size(0))
            top1_F1.update(F1_Score, input_L.size(0))

            if i % args.print_freq == 0:
                progress.display(i, log_txt_path)

        # TODO: this should also be done with the ProgressMeter
        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
            f.write('Current F1 Score: {top1_F1.avg:.4f}'.format(top1_F1=top1_F1) + '\n')
    return top1.avg, ls.avg,  top1_F1.avg


def save_checkpoint(state, is_best, checkpoint_path, best_checkpoint_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, log_txt_path):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)


def main():
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    now = datetime.datetime.now()
    time_str = now.strftime("[%m-%d]-[%H:%M]-")
    if args.is_face:
        extra_mess = args.dataset + '-' + args.data_mode + '-' + str(args.data_set) + time_str
    else:
        extra_mess = args.dataset + '-' + 'ori-' + args.data_mode + '-' + str(args.data_set) + time_str

    save_txt = 'result/' + extra_mess + '-' + time_str + '.txt'
    # save_txt = 'result/' + str(args.data_set) + 'train.txt'
    # 判断save_txt所在文件夹是否存在
    if not os.path.exists(os.path.dirname(save_txt)):
        os.makedirs(os.path.dirname(save_txt))
    with open(save_txt, "a+") as f:
        mes = args.mes + '\n' + str(args.bz) + ' ' + str(args.gamma) + ' ' + str(args.lr) + ' ' + str(
            args.epochs) + ' ' + str(
            args.mil) + '\n'
        f.write(mes)
    train_len = 2
    WAR_list = []
    UAR_list = []
    F1_list = []
    for i in range(1, train_len):
        # args.data_set = i
        train_data = train_data_loader(args.dataset, data_mode=args.data_mode, data_set=args.data_set,
                                       is_face=args.is_face, label_type=args.label_type)
        test_data = test_data_loader(args.dataset, data_mode=args.data_mode, data_set=args.data_set,
                                     is_face=args.is_face, label_type=args.label_type)

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=args.bz,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   drop_last=False)
        val_loader = torch.utils.data.DataLoader(test_data,
                                                 batch_size=32,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True)
        now = datetime.datetime.now()
        time_str = now.strftime("[%m-%d]-[%H:%M]-")
        project_path = './'

        log_txt_path = project_path + 'log/' + extra_mess + '-' + str(i) + '-log.txt'
        log_curve_path = project_path + 'log/' + extra_mess + '-' + str(i) + '-log.png'
        checkpoint_path = project_path + 'checkpoint/' + extra_mess + '-' + str(i) + '-model.pth'
        best_checkpoint_path = project_path + 'checkpoint/' + extra_mess + '-' + str(i) + '-model_best.pth'
        if not os.path.exists(os.path.dirname(best_checkpoint_path)):
            os.makedirs(os.path.dirname(best_checkpoint_path))
        best_acc, best_F1 = Run(args, train_loader, val_loader, log_txt_path, log_curve_path, checkpoint_path,
                       best_checkpoint_path)
        model = eth_net(args.n_in, args.n_embd, args.mlp_dim, args.max_len, args.arch, args.scale_factor, args.with_ln,
                        args.path_pdrop, args.downsample_type, args.thr_size, args.k, use_pos=args.use_pos,
                        num_classes=args.num_class)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        cudnn.benchmark = True

        UAR = test.validate_2(val_loader, model, args.num_class)
        txt = 'WAR: ' + str(best_acc.item()) + '\t' + 'UAR: ' + str(UAR) + '\t' + str(best_F1) + '\n'
        WAR_list.append(best_acc.item())
        UAR_list.append(UAR)
        F1_list.append(best_F1)
        with open(save_txt, "a+") as f:
            f.write(txt)
    with open(save_txt, "a+") as f:
        f.write('avg_WAR:' + str(sum(WAR_list) / (train_len - 1)) + '\t' + 'avg_UAR:' + str(
            sum(UAR_list) / (train_len - 1)) + '\t' + 'avg_F1:' + str(sum(F1_list) / (train_len - 1)) + '\n')


if __name__ == '__main__':
    main()
