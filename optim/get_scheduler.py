from torch.optim.lr_scheduler import StepLR


def get_scheduler(args, optimizer):
    scheduler_type = args.TRAIN.SCHEDULER_TYPE.lower()

    if scheduler_type == 'step_lr':
        scheduler = StepLR(optimizer, args.TRAIN.STEP_SIZE)
    else:
        raise NotImplementedError()

    return scheduler
