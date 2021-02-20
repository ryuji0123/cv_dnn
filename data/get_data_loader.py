from torch.utils.data import DataLoader


def getDataLoader(batch_size, dataset, sampler=None):
    return DataLoader(
            batch_size=batch_size, dataset=dataset, sampler=sampler,
            shuffle=False if sampler is not None else True
            )
