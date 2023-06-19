from torch import optim
from .util import LinearScheduler, CosineScheduler, ProportionScheduler
from .gam import GAM

def get_optim_and_schedulers(model, args):
    if args.base_opt == 'SGD':
        base_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
    elif args.base_opt == 'Adam':
        base_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    elif args.base_opt == 'AdamW':
        base_optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    else:
        raise NotImplementedError

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=args.epochs)

    # optional grad_rho_scheduler and grad_norm_rho_scheduler
    grad_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=lr_scheduler, max_lr=args.lr, min_lr=0.0,
                                             max_value=args.grad_rho, min_value=args.grad_rho)

    grad_norm_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=lr_scheduler, max_lr=args.lr, min_lr=0.0,
                                                  max_value=args.grad_norm_rho, min_value=args.grad_norm_rho)

    optimizer = GAM(params=model.parameters(), base_optimizer=base_optimizer, model=model,
                    grad_rho_scheduler=grad_rho_scheduler, grad_norm_rho_scheduler=grad_norm_rho_scheduler,
                    adaptive=args.adaptive, args=args)

    return optimizer, base_optimizer, lr_scheduler, grad_rho_scheduler, grad_norm_rho_scheduler