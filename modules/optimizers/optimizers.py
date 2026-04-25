import torch


def build_optimizer(args, model):

    params = filter(lambda x: id(x), model.parameters())
    optimizer = getattr(torch.optim, args.optim)(
        [ {'params': params, 'lr': args.lr}],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def build_lr_scheduler(args, optimizer):
    # lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience)
    return lr_scheduler

