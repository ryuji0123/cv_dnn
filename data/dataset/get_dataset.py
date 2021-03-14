import torchvision.transforms as transforms

from data.dataset.cifer_10_dataset import CIFAR10Dataset


def get_transform_from_list(transform_list: list):
    sequence = []

    for t in transform_list:
        if t == 'to_tensor':
            sequence.append(transforms.ToTensor())

        if t == 'normalize':
            sequence.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            )

    return transforms.Compose(sequence)


def get_dataset(args):
    transform = get_transform_from_list(args.DATA.TRANSFORM_LIST)
    dataset_type = args.DATA.DATASET_TYPE.lower()

    if dataset_type == 'cifer10':
        dataset = CIFAR10Dataset(
            root=args.DATA.CACHE_DIR,
            transform=transform,
            validation_size=args.DATA.VALIDATION_SIZE
        )

    else:
        raise NotImplementedError()

    return dataset
