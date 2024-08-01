# 融合多个模型的输出，eg. norm, rv, flow

import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.ETH_Net import eth_net
import matplotlib
from thop import profile
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

# import seaborn
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from dataloader.dataset import test_data_loader

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume_norm',
                    default='/home/lab/LM/ETH_Net/best_model/DFEW/DFEW-ori-norm-5-model_best.pth',
                    type=str, metavar='PATH', help='path to normal checkpoint')
parser.add_argument('--resume_rv',
                    default=None,
                    type=str, metavar='PATH', help='path to rv checkpoint')
parser.add_argument('--resume_flow',
                    default=None,
                    type=str, metavar='PATH', help='path to flow checkpoint')
parser.add_argument('--dataset', type=str, default='DFEW',
                    choices=['DFEW', 'FERv39k', 'AFEW', 'MAFW', 'CREMA-D', 'eNTERFACE05', 'RAVDESS'])
parser.add_argument('--label_type', type=str, default='single', choices=['single', 'compound'])
parser.add_argument('--num_class', type=int, default=7, choices=[6, 7, 8, 11, 43])
parser.add_argument('--is_face', type=bool, default=False)
parser.add_argument('--data_set', type=int, default=5)
parser.add_argument('--mu', type=float, default=0.9)

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

parser.add_argument('--gpu', type=str, default='3')

args = parser.parse_args()
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")
project_path = './'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# num_label = 7

# correct_pred = {str(classname): 0 for classname in range(num_label)}
# total_pred = {str(classname): 0 for classname in range(num_label)}

class_dict = {
    '0': 'Happiness',
    '1': 'Sadness',
    '2': 'Neutral',
    '3': 'Anger',
    '4': 'Surprise',
    '5': 'Disgust',
    '6': 'Fear'
}
emotions = ["hap", "sad", "neu", "ang", "sur", "dis", "fea"]

color_list = ['orange', 'blue', 'green', 'red', 'black', 'purple', 'pink', 'brown', 'gray', 'cyan', 'navy']


def plot_embedding(data, label):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    for i in range(data.shape[0]):
        # plt.text(data[i, 0], data[i, 1], str(label[i]),cd
        #          color=plt.cm.Set1(label[i] / 10.),
        #          fontdict={'weight': 'bold', 'size': 9})
        plt.plot(data[i, 0], data[i, 1], 'o', markersize=2, color=color_list[label[i]])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontproperties='Times New Roman', size=10)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontproperties='Times New Roman', size=10)
    if args.is_face:
        save_path = 'fig/' + args.dataset + '/' + str(args.data_set) + '.png'
    else:
        save_path = 'fig/' + args.dataset + '/' + 'ori-' + str(args.data_set) + '.png'

    # 判断save_path所在文件夹是否存在
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)


def main():
    print('The testing time: ' + now.strftime("%m-%d %H:%M"))
    print('The set: set ' + str(args.data_set))

    # create model and load pre_trained parameters
    model_norm = eth_net(args.n_in, args.n_embd, args.mlp_dim, args.max_len, args.arch, args.scale_factor, args.with_ln,
                         args.path_pdrop, args.downsample_type, args.thr_size, args.k, use_pos=args.use_pos,
                         num_classes=args.num_class)
    # input_4, input_8, input = torch.randn(1, 1408, 4), torch.randn(1, 1408, 8), torch.randn(1, 1408, 16)
    # mask_4, mask_8, mask = torch.ones(1, 1, 4), torch.ones(1, 1, 8), torch.ones(1, 1, 16)
    # flops, params = profile(model_norm, inputs=(input_4, input_8, input, mask_4, mask_8, mask, True))
    # print('the flops is {}G,the params is {}M'.format(round(flops / (10 ** 9), 2), round(params / (10 ** 6), 2)))2
    model_norm = torch.nn.DataParallel(model_norm).cuda()
    model_rv = eth_net(args.n_in, args.n_embd, args.mlp_dim, args.max_len, args.arch, args.scale_factor, args.with_ln,
                       args.path_pdrop, args.downsample_type, args.thr_size, args.k, use_pos=args.use_pos,
                       num_classes=args.num_class)
    model_rv = torch.nn.DataParallel(model_rv).cuda()
    model_flow = eth_net(args.n_in, args.n_embd, args.mlp_dim, args.max_len, args.arch, args.scale_factor, args.with_ln,
                         args.path_pdrop, args.downsample_type, args.thr_size, args.k, use_pos=args.use_pos,
                         num_classes=args.num_class)
    model_flow = torch.nn.DataParallel(model_flow).cuda()

    # optionally resume from a checkpoint
    if args.resume_norm:
        if os.path.isfile(args.resume_norm):
            print("=> loading checkpoint '{}'".format(args.resume_norm))
            checkpoint = torch.load(args.resume_norm)
            args.start_epoch = checkpoint['epoch']
            model_norm.load_state_dict(checkpoint['state_dict'])
            print("=> loaded norm checkpoint '{}' (epoch {})".format(args.resume_norm, checkpoint['epoch']))
        else:
            print("=> no norm checkpoint found at '{}'".format(args.resume_norm))

    if args.resume_rv:
        if os.path.isfile(args.resume_rv):
            print("=> loading checkpoint '{}'".format(args.resume_rv))
            checkpoint = torch.load(args.resume_rv)
            args.start_epoch = checkpoint['epoch']
            model_rv.load_state_dict(checkpoint['state_dict'])
            print("=> loaded rv checkpoint '{}' (epoch {})".format(args.resume_rv, checkpoint['epoch']))
        else:
            print("=> no rv checkpoint found at '{}'".format(args.resume_rv))

    if args.resume_flow:
        if os.path.isfile(args.resume_flow):
            print("=> loading checkpoint '{}'".format(args.resume_flow))
            checkpoint = torch.load(args.resume_flow)
            args.start_epoch = checkpoint['epoch']
            model_flow.load_state_dict(checkpoint['state_dict'])
            print("=> loaded flow checkpoint '{}' (epoch {})".format(args.resume_flow, checkpoint['epoch']))
        else:
            print("=> no flow checkpoint found at '{}'".format(args.resume_flow))

    cudnn.benchmark = True

    # Data loading code
    val_data_norm = test_data_loader(dataset=args.dataset, data_mode='norm', data_set=args.data_set,
                                     is_face=args.is_face,
                                     label_type=args.label_type)

    val_loader_norm = torch.utils.data.DataLoader(val_data_norm,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)
    val_data_rv = test_data_loader(dataset=args.dataset, data_mode='rv', data_set=args.data_set, is_face=args.is_face,
                                   label_type=args.label_type)
    val_loader_rv = torch.utils.data.DataLoader(val_data_rv,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
    # val_data_flow = test_data_loader(data_set=args.data_set, data_mode='flow')
    # val_loader_flow = torch.utils.data.DataLoader(val_data_flow,
    #                                               batch_size=args.batch_size,
    #                                               shuffle=False,
    #                                               num_workers=args.workers,
    #                                               pin_memory=True)

    if args.resume_rv and args.resume_flow:
        result, tar, c_m = validate(val_loader_norm, val_loader_rv, None, model_norm, model_rv, None, mode=2,
                                    num_label=args.num_class)
    elif args.resume_rv:
        result, tar, c_m = validate(val_loader_norm, val_loader_rv, None, model_norm, model_rv, None, mode=1,
                                    num_label=args.num_class)
    else:
        result, tar, c_m = validate(val_loader_norm, None, None, model_norm, None, None, mode=0,
                                    num_label=args.num_class)

    # t-SNE
    # if args.label_type == 'single':
    #     tsne = TSNE(n_components=2, init='pca', random_state=0)
    #     result = tsne.fit_transform(result)
    #     plot_embedding(result, tar)

    # Confusion matrix
    # CM = []
    # for row in c_m:
    #     row = row / np.sum(row)
    #     CM.append(row * 100)
    # ax = seaborn.heatmap(
    #     CM, xticklabels=emotions, yticklabels=emotions, cmap='rocket_r', annot=True, fmt='.2f')
    # figure = ax.get_figure()
    # # save the heatmap
    # if args.is_face:
    #     save_path = 'fig/' + args.dataset + '/' + "cm-" + str(args.data_set) + '.png'
    # else:
    #     save_path = 'fig/' + args.dataset + '/' + "cm-" + 'ori-' + str(args.data_set) + '.png'
    # figure.savefig(save_path)
    # plt.close()


def accuracy_war(output, target, topk=(1,)):
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


def accuracy_uar(output, target, correct_pred, total_pred, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred_top1 = pred[:, 0]
        for label, prediction in zip(target, pred_top1):
            if label == prediction:
                correct_pred[str(prediction.detach().cpu().numpy())] += 1
            total_pred[str(label.detach().cpu().numpy())] += 1
        return correct_pred, total_pred


def validate(val_loader_norm=None, val_loader_rv=None, val_loader_flow=None, model_norm=None, model_rv=None,
             model_flow=None, mode=2, num_label=7):
    correct_pred = {str(classname): 0 for classname in range(num_label)}
    total_pred = {str(classname): 0 for classname in range(num_label)}
    all_pred, all_target = [], []
    top1 = AverageMeter('Accuracy', ':6.3f')
    # progress = ProgressMeter(len(val_loader),
    #                          [top1],
    #                          prefix='Test: ')

    # switch to evaluate mode
    Result = list()
    Target = list()
    with torch.no_grad():
        if mode == 2:
            model_norm.eval()
            model_rv.eval()
            model_flow.eval()
            for i, (input_H, input_M, input_L, target) in enumerate(
                    zip(val_loader_norm, val_loader_rv, val_loader_flow)):
                input_L, input_M, input_H, masks_L, masks_M, masks_H = input_L[0][0].cuda(), input_M[0][0].cuda(), \
                    input_H[
                        0][0].cuda(), \
                    input_L[0][1].cuda(), input_M[0][1].cuda(), input_H[0][1].cuda()
                pred_norm = model_norm(None, None, input_H, None, None, masks_H, False)

                input_L, input_M, input_H, masks_L, masks_M, masks_H = input_L[1][0].cuda(), input_M[1][0].cuda(), \
                    input_H[1][0].cuda(), input_L[1][1].cuda(), input_M[1][1].cuda(), input_H[1][1].cuda()
                pred_rv = model_rv(None, None, input_H, None, None, masks_H, False)

                input_L, input_M, input_H, masks_L, masks_M, masks_H = input_L[2][0].cuda(), input_M[2][0].cuda(), \
                    input_H[2][0].cuda(), input_L[2][1].cuda(), input_M[2][1].cuda(), input_H[2][1].cuda()
                pred_flow = model_flow(None, None, input_H, None, None, masks_H, False)

                pred = pred_norm + pred_rv + pred_flow
                Result.extend(pred.detach().cpu().numpy())
                # measure accuracy and record loss
                Target.extend(target[0].numpy())
                target = target.cuda()
                acc1, _ = accuracy_uar(pred, target, correct_pred, total_pred, topk=(1, 5))
                top1.update(acc1[0], input_H.size(0))
        elif mode == 1:
            model_norm.eval()
            model_rv.eval()
            # 同时遍历val_loader_norm和val_loader_rv
            for i, (
            (input_H_norm, input_M_norm, input_L_norm, target_norm), (input_H_rv, input_M_rv, input_L_rv, target_rv)) \
                    in enumerate(zip(val_loader_norm, val_loader_rv)):
                input_L_norm, input_M_norm, input_H_norm, masks_L_norm, masks_M_norm, masks_H_norm = input_L_norm[
                    0].cuda(), input_M_norm[0].cuda(), \
                    input_H_norm[0].cuda(), input_L_norm[1].cuda(), input_M_norm[1].cuda(), input_H_norm[1].cuda()
                pred_norm = model_norm(None, None, input_H_norm, None, None, masks_H_norm, False)
                input_L_rv, input_M_rv, input_H_rv, masks_L_rv, masks_M_rv, masks_H_rv = input_L_rv[0].cuda(), \
                input_M_rv[0].cuda(), \
                    input_H_rv[0].cuda(), input_L_rv[1].cuda(), input_M_rv[1].cuda(), input_H_rv[1].cuda()
                pred_rv = model_rv(None, None, input_H_rv, None, None, masks_H_rv, False)
                pred = args.mu * pred_norm + (1 - args.mu) * pred_rv

                Result.extend(pred.detach().cpu().numpy())
                # measure accuracy and record loss
                Target.extend(target_norm.numpy())
                target = target_norm.cuda()
                acc1, _ = accuracy_war(pred, target, topk=(1, 5))
                correct_pred, total_pred = accuracy_uar(pred, target, correct_pred, total_pred, topk=(1, 5))
                top1.update(acc1[0], input_H_norm.size(0))

                pred = torch.argmax(pred, 1).cpu().detach().numpy()
                target = target.cpu().numpy()

                all_pred.extend(pred)
                all_target.extend(target)
        else:
            model_norm.eval()
            file_data = ""
            for i, (input_H, input_M, input_L, target) in enumerate(val_loader_norm):
                input_L, input_M, input_H, masks_L, masks_M, masks_H = input_L[0].cuda(), input_M[0].cuda(), input_H[
                    0].cuda(), \
                    input_L[1].cuda(), input_M[1].cuda(), input_H[1].cuda()
                pred = model_norm(None, None, input_H, None, None, masks_H, False)
                Result.extend(pred.detach().cpu().numpy())
                # measure accuracy and record loss
                Target.extend(target.numpy())
                target = target.cuda()
                acc1, _ = accuracy_war(pred, target, topk=(1, 5))
                correct_pred, total_pred = accuracy_uar(pred, target, correct_pred, total_pred, topk=(1, 5))
                top1.update(acc1[0], input_H.size(0))

        c_m = confusion_matrix(all_target, all_pred)

        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        # with open('result/mes.txt', "w", encoding="utf-8") as f:
        #     f.write(file_data)
        cur_list = []
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            # cur_list.append("{:.2f}".format(accuracy))
            cur_list.append(accuracy)
        avg_UAR = sum(cur_list) / num_label
        avg_WAR = top1.avg
        result = ''
        for i, cur_acc in enumerate(cur_list):
            result += str(cur_acc) + '\t'
        print('each acc: ', result)
        print("avg_WAR: ", avg_WAR.item())
        print("avg_UAR: ", avg_UAR)
        result = Result[0]
        tar = Target[0]
        for i in range(1, len(Result)):
            result = np.vstack((result, Result[i]))
            tar = np.vstack((tar, Target[i]))
        tar = tar.reshape(-1)
    return result, tar, c_m


def validate_1(val_loader_norm=None, val_loader_rv=None, val_loader_flow=None, model_norm=None, model_rv=None,
               model_flow=None, mode=2, num_label=7):
    WAR_list = ""
    UAR_list = ""

    for j in range(11):
        print("*************************" + str(j) + "*******************")
        correct_pred = {str(classname): 0 for classname in range(num_label)}
        total_pred = {str(classname): 0 for classname in range(num_label)}
        all_pred, all_target = [], []
        top1 = AverageMeter('Accuracy', ':6.3f')
        # progress = ProgressMeter(len(val_loader),
        #                          [top1],
        #                          prefix='Test: ')

        # switch to evaluate mode
        Result = list()
        Target = list()
        with torch.no_grad():
            model_norm.eval()
            model_rv.eval()
            # 同时遍历val_loader_norm和val_loader_rv
            for i, (
            (input_H_norm, input_M_norm, input_L_norm, target_norm), (input_H_rv, input_M_rv, input_L_rv, target_rv)) \
                    in enumerate(zip(val_loader_norm, val_loader_rv)):
                input_L_norm, input_M_norm, input_H_norm, masks_L_norm, masks_M_norm, masks_H_norm = input_L_norm[
                    0].cuda(), input_M_norm[0].cuda(), \
                    input_H_norm[0].cuda(), input_L_norm[1].cuda(), input_M_norm[1].cuda(), input_H_norm[1].cuda()
                pred_norm = model_norm(None, None, input_H_norm, None, None, masks_H_norm, False)
                input_L_rv, input_M_rv, input_H_rv, masks_L_rv, masks_M_rv, masks_H_rv = input_L_rv[0].cuda(), \
                input_M_rv[0].cuda(), \
                    input_H_rv[0].cuda(), input_L_rv[1].cuda(), input_M_rv[1].cuda(), input_H_rv[1].cuda()
                pred_rv = model_rv(None, None, input_H_rv, None, None, masks_H_rv, False)
                mu = 0.1 * j
                pred = mu * pred_norm + (1 - mu) * pred_rv
                Result.extend(pred.detach().cpu().numpy())
                # measure accuracy and record loss
                Target.extend(target_norm.numpy())
                target = target_norm.cuda()
                acc1, _ = accuracy_war(pred, target, topk=(1, 5))
                correct_pred, total_pred = accuracy_uar(pred, target, correct_pred, total_pred, topk=(1, 5))
                top1.update(acc1[0], input_H_norm.size(0))

                pred = torch.argmax(pred, 1).cpu().detach().numpy()
                target = target.cpu().numpy()

                all_pred.extend(pred)
                all_target.extend(target)

            # WAR
            acc1 = accuracy_score(all_target, all_pred)
            # UAR
            acc2 = balanced_accuracy_score(all_target, all_pred)
            c_m = confusion_matrix(all_target, all_pred)

            print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
            WAR_list += str(acc1) + '\t'
            cur_list = []
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                # cur_list.append("{:.2f}".format(accuracy))
                cur_list.append(accuracy)
            avg_UAR = sum(cur_list) / num_label
            avg_WAR = top1.avg
            result = ''
            for i, cur_acc in enumerate(cur_list):
                result += str(cur_acc) + '\t'
            print('each acc: ', result)
            print("avg_WAR: ", acc2)
            print("avg_UAR: ", avg_UAR)
            UAR_list += str(acc2) + '\t'
            result = Result[0]
            tar = Target[0]
            for i in range(1, len(Result)):
                result = np.vstack((result, Result[i]))
                tar = np.vstack((tar, Target[i]))
            tar = tar.reshape(-1)

    print(WAR_list)
    print(UAR_list)
    return result, tar, c_m


def validate_2(val_loader, model, num_label=7):
    correct_pred = {str(classname): 0 for classname in range(num_label)}
    total_pred = {str(classname): 0 for classname in range(num_label)}
    top1 = AverageMeter('Accuracy', ':6.3f')
    model.eval()

    with torch.no_grad():
        Result = list()
        Target = list()
        for i, (input_H, input_M, input_L, target) in enumerate(val_loader):
            input_L, input_M, input_H, masks_L, masks_M, masks_H = input_L[0].cuda(), input_M[0].cuda(), input_H[
                0].cuda(), \
                input_L[1].cuda(), input_M[1].cuda(), input_H[1].cuda()
            Target.extend(target.numpy())
            target = target.cuda()

            # compute output
            pred = model(None, None, input_H, None, None, masks_H, False)
            Result.extend(pred.detach().cpu().numpy())
            # measure accuracy and record loss
            acc1, _ = accuracy_war(pred, target, topk=(1, 5))
            correct_pred, total_pred = accuracy_uar(pred, target, correct_pred, total_pred, topk=(1, 5))
            top1.update(acc1[0], input_H.size(0))

        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        cur_list = []
        for classname, correct_count in correct_pred.items():
            if total_pred[classname] == 0:
                accuracy = 100
            else:
                accuracy = 100 * float(correct_count) / total_pred[classname]
            # cur_list.append("{:.2f}".format(accuracy))
            cur_list.append(accuracy)
        avg_UAR = sum(cur_list) / num_label
        result = ''
        for i, cur_acc in enumerate(cur_list):
            result += str(cur_acc) + '\t'
        print(result)
        print(avg_UAR)
    return avg_UAR


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

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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


if __name__ == '__main__':
    main()
