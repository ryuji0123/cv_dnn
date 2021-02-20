from torch.optim import Adam, SGD


def getOptimizer(args, model):
    optimizer_set = set(['adam', 'sgd'])
    optimizer_type = args.TRAIN.OPTIMIZER_TYPE.lower()
    if optimizer_type not in optimizer_set:
        raise NotImplementedError()

    if optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=args.TRAIN.LR)
    elif optimizer_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.TRAIN.LR, momentum=args.TRAIN.MOMENTUM)

    return optimizer
