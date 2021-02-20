from trainer.simple_cnn_trainer import SimpleCNNTrainer


def getTrainer(args):
    trainer_dict = {
            'simple_cnn': SimpleCNNTrainer,
            }

    model_type = args.TRAIN.MODEL_TYPE.lower()
    if model_type not in trainer_dict:
        raise NotImplementedError()

    return trainer_dict[model_type]()
