from torch.optim import Adam, SGD


def get_optimizer(args, model):
    optimizer_type = args.TRAIN.OPTIMIZER_TYPE.lower()

    if optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=args.TRAIN.LR)
    elif optimizer_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.TRAIN.LR, momentum=args.TRAIN.MOMENTUM)
    else:
        raise NotImplementedError()

    return optimizer
