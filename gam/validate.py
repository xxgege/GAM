import time
import torch

import torch.nn.parallel
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from utilis.meters import AverageMeter
from utilis.meters import ProgressMeter
from utilis.matrix import accuracy


def validate(gpu, val_loader, model, criterion, test=True, args=None):
    if test:
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')
    else:
        batch_time = AverageMeter('val Time', ':6.3f')
        losses = AverageMeter('val Loss', ':.4e')
        top1 = AverageMeter('Val Acc@1', ':6.2f')
        top5 = AverageMeter('Val Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Val: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

            if args.dataset.startswith('CIFAR'):
                output = model(images)
            else:
                output, cfeatures = model(images)
            loss = criterion(output, target)

            acc1 = accuracy(output, target, topk=(1, ))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                method_name = args.log_path.split('/')[-2]
                progress.display(i, method_name)
                progress.write_log(i, args.log_path)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        with open(args.log_path, 'a') as f1:
            f1.writelines(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                          .format(top1=top1, top5=top5))

    return top1.avg
