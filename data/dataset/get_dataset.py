import torchvision.transforms as transforms

from data.dataset.cifer_10_dataset import CIFAR10Dataset


def getTransformFromList(transform_list):
    sequence = []

    for t in transform_list:
        if t == 'to_tensor':
            sequence.append(transforms.ToTensor())

        if t == 'normalize':
            sequence.append(
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    )

    return transforms.Compose(sequence)


def getDataset(args):
    dataset_ref = set(['cifer10'])

    dataset_type = args.DATA.DATASET_TYPE.lower()
    if dataset_type not in dataset_ref:
        raise NotImplementedError()

    transform = getTransformFromList(args.DATA.TRANSFORM_LIST)

    if dataset_type == 'cifer10':
        dataset = CIFAR10Dataset(
            root=args.DATA.CACHE_DIR, transform=transform, val_size=args.DATA.VAL_SIZE
            )

    return dataset
