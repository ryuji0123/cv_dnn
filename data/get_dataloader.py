from torch.utils.data import DataLoader, Dataset


def get_dataloader(
    batch_size: int,
    dataset: Dataset,
    num_workers: int = 2,
    sampler=None,
) -> DataLoader:
    return DataLoader(
        batch_size=batch_size,
        dataset=dataset,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=False if sampler is not None else True
    )
