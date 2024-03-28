import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils import pcl_loader
from models import moco_encoder
from main_kmeans import run_kmeans
from utils.meter_functions import AverageMeter, ProgressMeter, accuracy
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/app/dataset/ND/Full/NestedUVC_DualAttention_Parallel_Fourier_MSE', 
                    help='path to datset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', 
                    help='backbone model')

parser.add_argument('--epochs', default=200, type=int, metavar='N', 
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', 
                    help='mini-batch size (default: 24)')
parser.add_argument('--lr', '--learning_rate', default=0.03, type=float, metavar='LR', 
                    help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, 
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', 
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', 
                    help='weight dcay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', 
                    help='print frequency interval (fault: 10)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: ./PCL_v2_epoch200.pth.tar)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', 
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--seed', default=1004, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default='cuda:0',
                    help='GPU id to use.')

parser.add_argument('--low-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--pcl-r', default=16384, type=int,
                    help='queue size; number of negative pairs; needs to be smaller than num_cluster (default: 16384)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature')

parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco-v2/SimCLR data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--num-cluster', default='50,100,200', type=str, 
                    help='number of clusters')
parser.add_argument('--warmup-epoch', default=20, type=int,
                    help='number of warm-up epochs to only train with InfoNCE loss')
parser.add_argument('--exp_dir', default='/app/3rd/backup/PCL', type=str,
                    help='experiment directory')
parser.add_argument('--exp_log_dir', default='/app/3rd/backup/PCL/log', type=str,
                    help='experiment directory')
parser.add_argument('--exp_ckp_dir', default='/app/3rd/backup/PCL/ckp', type=str,
                    help='experiment directory')



def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    
    args.num_cluster = args.num_cluster.split(',')

    # 실험 결과 저장
    os.makedirs(args.exp_log_dir, exist_ok=True)
    os.makedirs(args.exp_ckp_dir, exist_ok=True)

    main_worker(args.gpu, args)

    
def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    
    # Create Momentum Encoder Model
    print(f"Creating Momentum Encoder Model ==> {args.arch}")
    model = moco_encoder.MoCo(models.__dict__[args.arch],
                         args.low_dim,
                         args.pcl_r,
                         args.moco_m,
                         args.temperature,
                         args.mlp)
    
    # print(model)

    if args.gpu is not None:
        model = model.to(args.gpu)

    
    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.gpu)
    
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay
                                )
    
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print(f'Loading pretrained checkpoint ==> [{args.pretrained}] ...')
            checkpoint = torch.load(args.pretrained, map_location=args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"loaded pretrained checkpoint ==> {args.pretrained}")
        
        else:
            print(f"No pretrained checkpoint found at [{args.pretrained}]")

    if args.resume:
        if os.path.isfile(args.resume):
            print(f'Loading checkpoint ==> [{args.resume}] ...')
            checkpoint = torch.load(args.resume, map_location=args.gpu)

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print(f"loaded checkpoint ==> {args.resume}")

        else:
            print(f"No checkpoint found at [{args.resume}]")

    
    # Data loading code
    traindir = f'{args.data}/1-fold/B'
    evaldir = f'{args.data}/2-fold/A'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([pcl_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        # center-crop augmentation 
    eval_augmentation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
        ])
    
    train_dataset = pcl_loader.ImageFolderInstance(
        traindir,
        pcl_loader.TwoCropsTransform(transforms.Compose(augmentation)))
    
    eval_dataset = pcl_loader.ImageFolderInstance(
        evaldir,
        eval_augmentation)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size*5, shuffle=False)
    
    summary = SummaryWriter(f"{args.exp_log_dir}/log")
    for epoch in range(args.start_epoch, args.epochs):

        cluster_result = None
        if epoch >= args.warmup_epoch:
            # compute momentum features for center-cropped images
            features = compute_features(eval_loader, model, args)

            # placeholder for clustering result
            cluster_result = {'img2cluster': [], 'centroids': [], 'density': []}
            for num_cluster in args.num_cluster:
                cluster_result['img2cluster'].append(torch.zeros(len(eval_dataset), dtype=torch.long).to(args.gpu))
                cluster_result['centroids'].append(torch.zeros(int(num_cluster),args.low_dim).to(args.gpu))
                cluster_result['density'].append(torch.zeros(int(num_cluster)).to(args.gpu)) 


            features[torch.norm(features, dim=1) > 1.5] /= 2 # account for the few samples that are computed twice
            features = features.numpy()
            cluster_result = run_kmeans(features, args)
            
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        losses, acc_inst, acc_proto= train(train_loader, model, criterion, optimizer, epoch, args, cluster_result)

        summary.add_scalar('Train/loss', losses, epoch)
        summary.add_scalar('Val/acc_inst', acc_inst, epoch)
        summary.add_scalar('Val/acc_proto', acc_proto, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename=f'{args.exp_ckp_dir}/checkpoint_{epoch:04d}.pth.tar')
        
        # if (epoch + 1) % 5 == 0:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }, is_best=False, filename=f'{args.exp_ckp_dir}/checkpoint_{epoch:04d}.pth.tar')
                

def train(train_loader, model, criterion, optimizer, epoch, args, cluster_result=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')   
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst, acc_proto],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, index) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            img[0] = img[0].to(args.gpu)
            img[1] = img[1].to(args.gpu)

        # compute output
        output, target, output_proto, target_proto = model(im_q=img[0], im_k=img[1], cluster_result=cluster_result, index=index)
        # print(f'output: {output} | target: {target} | output_proto: {output_proto} | target_proto: {target_proto}')

        # InfoNCE loss
        loss = criterion(output, target)

        # ProtoNCE loss
        if output_proto is not None:
            loss_proto = 0
            for proto_out, proto_target in zip(output_proto, target_proto):
                loss_proto += criterion(proto_out, proto_target)
                accp = accuracy(proto_out, proto_target)[0]
                acc_proto.update(accp[0], img[0].size(0))

            # average loss across all sets of prototypes
            loss_proto /= len(args.num_cluster)
            loss += loss_proto

        losses.update(loss.item(), img[0].size(0))
        acc = accuracy(output, target)[0] 
        acc_inst.update(acc[0], img[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.get_avg(), acc_inst.get_avg(), acc_proto.get_avg()



def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset), args.low_dim).to(args.gpu)

    for _, (img, idx) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            img = img.to(args.gpu)
            feat = model(img, is_eval=True)
            features[idx] = feat

        return feat.cpu()
    

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()