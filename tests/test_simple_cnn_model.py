from config import update_args
from data import get_dataloader
from data.dataset import get_dataset
from models import get_model


def test_simple_cnn_model():
    args = update_args(cfg_file='simple_cnn')
    device = 'cpu'

    train_data_dict = get_dataset(args).train_data_dict

    train_dataloader = get_dataloader(
        batch_size=args.TRAIN.BATCH_SIZE,
        dataset=train_data_dict['dataset'],
        num_workers=args.DATA.NUM_WORKERS,
        sampler=train_data_dict['train_sampler'],
    )

    validation_dataloader = get_dataloader(
        batch_size=args.TRAIN.BATCH_SIZE,
        dataset=train_data_dict['dataset'],
        num_workers=args.DATA.NUM_WORKERS,
        sampler=train_data_dict['validation_sampler'],
    )

    model = get_model(
        args=args,
        device=device,
        hparams={
            'learning rate': args.TRAIN.LR,
            'batch size': args.TRAIN.BATCH_SIZE,
        },
    )

    for data, labels in train_dataloader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        assert data.shape[0] == labels.shape[0] == outputs.shape[0] == args.TRAIN.BATCH_SIZE
        assert data.shape[1] == 3
        assert outputs.shape[1] == len(args.DATA.CLASSES)

        break

    for data, labels in validation_dataloader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        assert data.shape[0] == labels.shape[0] == outputs.shape[0] == args.TRAIN.BATCH_SIZE
        assert data.shape[1] == 3
        assert outputs.shape[1] == len(args.DATA.CLASSES)

        break
