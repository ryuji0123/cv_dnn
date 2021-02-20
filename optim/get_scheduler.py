from  torch.optim.lr_scheduler import StepLR


def getScheduler(args, optimizer):
    scheduler_set = set(['step_lr'])
    scheduler_type = args.TRAIN.SCHEDULER_TYPE.lower()
    if scheduler_type not in scheduler_set:
        raise NotImplementedError()

    if scheduler_type == 'step_lr':
        scheduler = StepLR(optimizer, args.TRAIN.STEP_SIZE)

    return scheduler
