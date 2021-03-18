from torch.optim import Adam, Optimizer, SGD


def get_optimizer(args, model, lr: float = None) -> Optimizer:
    optimizer_type = args.TRAIN.OPTIMIZER_TYPE.lower()
    lr = args.TRAIN.LR if lr is None else lr

    if optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, momentum=args.TRAIN.MOMENTUM)
    else:
        raise NotImplementedError()

    return optimizer
