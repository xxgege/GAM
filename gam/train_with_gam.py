import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utilis.meters import AverageMeter
from utilis.meters import ProgressMeter
from utilis.matrix import accuracy

from .util import ProportionScheduler
from .gam import GAM
from .smooth_cross_entropy import smooth_crossentropy


def train_epoch(gpu, train_loader, model, base_optimizer, epoch, args,
                lr_scheduler=None, grad_rho_scheduler=None,
                grad_norm_rho_scheduler=None, optimizer=None, ):
    '''
        To train the model with gam
    '''
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    Lr = AverageMeter('Lr', ':.4e')

    progress = ProgressMeter(
        len(train_loader),
        [losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    lr = base_optimizer.param_groups[0]['lr']

    if not grad_rho_scheduler and not grad_norm_rho_scheduler and not optimizer:
        grad_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=lr_scheduler, max_lr=args.lr, min_lr=0.0,
                                                 max_value=args.grad_rho_max, min_value=args.grad_rho_min)

        grad_norm_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=lr_scheduler, max_lr=args.lr, min_lr=0.0,
                                                      max_value=args.grad_norm_rho_max,
                                                      min_value=args.grad_norm_rho_min)

        optimizer = GAM(params=model.parameters(), base_optimizer=base_optimizer, model=model,
                        grad_rho_scheduler=grad_rho_scheduler, grad_norm_rho_scheduler=grad_norm_rho_scheduler,
                        adaptive=args.adaptive, args=args)

    def loss_fn(predictions, targets):
        return smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing).mean()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

        optimizer.set_closure(loss_fn, images, target)

        predictions, loss = optimizer.step()

        with torch.no_grad():
            optimizer.update_rho_t()
            acc1, acc5 = accuracy(predictions, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            losses.update(loss.item(), images.size(0))

        optimizer.zero_grad()

        Lr.update(lr, 1)
        method_name = args.log_path.split('/')[-2]

        if i % args.print_freq == 0:
            progress.display(i, method_name)
            progress.write_log(i, args.log_path)

    if torch.isnan(loss).any():
        raise SystemExit('NaN！')
