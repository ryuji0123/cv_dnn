from model.simple_cnn_model import SimpleCNNModel


def getModel(args):
    model_dict = set([
        'simple_cnn'
        ])

    model_type = args.TRAIN.MODEL_TYPE.lower()
    if model_type not in model_dict:
        raise NotImplementedError()

    if model_type == 'simple_cnn':
        model = SimpleCNNModel(
                in_channel=args.DATA.INPUT_DIM,
                out_channel=len(args.DATA.CLASSES)
                )

        return model
