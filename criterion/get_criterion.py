from torch.nn import CrossEntropyLoss


def getCriterion(args):
    criterion_dict = {
            'cross_entropy': CrossEntropyLoss,
            }
    
    criterion_type = args.TRAIN.CRITERION_TYPE.lower()
    if criterion_type not in criterion_dict:
        raise NotImplementedError()

    return criterion_dict[criterion_type]()
