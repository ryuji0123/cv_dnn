import pytorch_lightning as pl
import torch

from contextlib import redirect_stdout
from shutil import rmtree

from mlflow.tracking.client import MlflowClient
from pytorch_lightning.loggers import MLFlowLogger

from config import updateArgs, parseConsole
from data import getDataLoader
from data.dataset import getDataset
from model import getModel
from optim import getOptimizer, getScheduler


def main(args, args_file_path, tmp_results_dir, train_log_file_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_dict = getDataset(args).train_data

    train_data_loader = getDataLoader(
            batch_size=args.TRAIN.BATCH_SIZE,
            dataset=train_data_dict['dataset'],
            num_workers=args.DATA.NUM_WORKERS,
            sampler=train_data_dict['train_sampler']
            )

    val_data_loader = getDataLoader(
            batch_size=args.TRAIN.BATCH_SIZE,
            dataset=train_data_dict['dataset'],
            num_workers=args.DATA.NUM_WORKERS,
            sampler=train_data_dict['val_sampler']
            )

    model = getModel(
        args=args,
        device=device,
        ).to(device)

    mlflow_logger = MLFlowLogger(
            experiment_name=args.MLFLOW.EXPERIMENT_NAME,
            )

    trainer = pl.Trainer(
            distributed_backend=args.TRAIN.DISTRIBUTED_BACKEND,
            gpus=args.TRAIN.GPUS,
            logger=mlflow_logger,
            max_epochs=args.TRAIN.MAX_EPOCHS,
            )

    try:
        trainer.fit(model, train_data_loader, val_data_loader)
    finally:
        with open(args_file_path, 'w') as f:
            with redirect_stdout(f):
                print(args.dump())

        mlflow_client = MlflowClient()
        mlflow_client.log_artifact(mlflow_logger.run_id, args_file_path)
        mlflow_client.log_artifact(mlflow_logger.run_id, train_log_file_path)
        rmtree(tmp_results_dir)


if __name__ == '__main__':
    option = parseConsole()
    args = updateArgs(cfg_file=option.cfg_file_path)
    main(
            args=args,
            args_file_path=option.args_file_path,
            tmp_results_dir=option.tmp_results_dir,
            train_log_file_path=option.train_log_file_path
            )
