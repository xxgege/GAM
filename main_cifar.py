import argparse
import os
import random
import warnings
import models
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
from torch.utils.tensorboard import SummaryWriter

from gam.validate import validate
from gam.train_with_gam import train_epoch
from gam.optimizer_helper import get_optim_and_schedulers

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

print('model name space', model_names)
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training with GAM')

# model & training
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_c',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--epochs_decay', default=[30, 60], type=int,
                    help='seed for initializing training. ')

parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--benchmark', default=True, type=bool,
                    help='GPU id to use.')

# data
parser.add_argument('--cifar10_path', metavar='DIR_C', default='./data/cifar10',
                    help='path to dataset')
parser.add_argument('--cifar100_path', metavar='DIR_C', default='./data/cifar100',
                    help='path to dataset')
parser.add_argument('--dataset', default='CIFAR100', type=str)

parser.add_argument('--log_base',
                    default='./results', type=str, metavar='PATH',
                    help='path to save logs (default: none)')

# opt
parser.add_argument("--base_opt", default='SGD', type=str, help="")

parser.add_argument("--grad_beta_0", default=1., type=float, help="scale for g0")
parser.add_argument("--grad_beta_1", default=1., type=float, help="scale for g1")
parser.add_argument("--grad_beta_2", default=-1., type=float, help="scale for g2")
parser.add_argument("--grad_beta_3", default=1., type=float, help="scale for g3")

# parser.add_argument("--grad_rho_max", default=0.04, type=int, help="")
# parser.add_argument("--grad_rho_min", default=0.02, type=int, help="")
parser.add_argument("--grad_rho", default=0.02, type=int, help="")

# parser.add_argument("--grad_norm_rho_max", default=0.04, type=int, help="")
# parser.add_argument("--grad_norm_rho_min", default=0.02, type=int, help="")
parser.add_argument("--grad_norm_rho", default=0.2, type=int, help="")

parser.add_argument("--adaptive", default=False, type=bool, help="")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")

parser.add_argument("--grad_gamma", default=0.03, type=int, help="")

return_acc = 0

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def main():
    torch.backends.cudnn.benchmark = True
    args = parser.parse_args()

    # default hps
    if args.dataset == 'CIFAR100':
        if args.arch.startswith('resnet'):
            args.grad_norm_rho, args.grad_rho, args.grad_beta_0, args.grad_beta_1, args.grad_gamma = 0.2, 0.02, 0.5, 0.6, 0.03
        elif args.arch.startswith('pyramidnet'):
            args.grad_norm_rho, args.grad_rho, args.grad_beta_0, args.grad_beta_1, args.grad_gamma = 0.2, 0.04, 0.3, 0.5, 0.05
    elif args.dataset == 'CIFAR10':
        if args.arch.startswith('resnet'):
            args.grad_norm_rho, args.grad_rho, args.grad_beta_0, args.grad_beta_1, args.grad_gamma = 0.2, 0.03, 0.1, 0.1, 0.05
        elif args.arch.startswith('pyramidnet'):
            args.grad_norm_rho, args.grad_rho, args.grad_beta_0, args.grad_beta_1, args.grad_gamma = 0.2, 0.03, 0.1, 0.1, 0.03

    args.grad_beta_2 = 1 - args.grad_beta_0
    args.grad_beta_3 = 1 - args.grad_beta_1

    log_description = 'GAM'
    args.log_path = os.path.join(args.log_base, args.dataset, log_description, "log.txt")

    if args.seed is not None:
        # for reimplement
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global return_acc
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    num_classes = 10 if args.dataset == 'CIFAR10' else 100
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, num_classes=num_classes)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=num_classes)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    cudnn.benchmark = args.benchmark

    if args.dataset == 'CIFAR10':
        data_root = args.cifar10_path
        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.2023, 0.1994, 0.2010])
            ]))

        test_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.2023, 0.1994, 0.2010])
            ]))

    else:
        data_root = args.cifar100_path
        train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ]))

        test_dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    log_dir = os.path.dirname(args.log_path)
    print('tensorboard dir {}'.format(log_dir))
    tensor_writer = SummaryWriter(log_dir)

    # get base opt and schedulers
    optimizer, base_optimizer, lr_scheduler, grad_rho_scheduler, grad_norm_rho_scheduler = get_optim_and_schedulers(
        model, args)

    for epoch in range(args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_epoch(gpu, train_loader, model, base_optimizer, epoch, args,
                    lr_scheduler=lr_scheduler, grad_rho_scheduler=grad_rho_scheduler,
                    grad_norm_rho_scheduler=grad_norm_rho_scheduler, optimizer=optimizer)

        lr_scheduler.step()

        acc1 = validate(gpu, val_loader, model, criterion, True, args)
        return_acc = max(return_acc, acc1)
        tensor_writer.add_scalar('return_ACC@1/test', return_acc, epoch)

    print('Test top-1 acc: ', return_acc)


if __name__ == '__main__':
    main()
