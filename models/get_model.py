from pytorch_lightning import LightningModule

from models.simple_cnn_model import SimpleCNNModel


def get_model(args, device) -> LightningModule:
    model_set = set([
        'simple_cnn'
        ])

    model_type = args.TRAIN.MODEL_TYPE.lower()

    if model_type == 'simple_cnn':
        model = SimpleCNNModel(
            args=args,
            device=device,
            in_channel=args.DATA.INPUT_DIM,
            out_channel=len(args.DATA.CLASSES),
        )
    else:
        raise NotImplementedError()

    return model
