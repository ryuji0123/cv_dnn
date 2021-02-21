import torch

from config import updateArgs, parseConsole
from criterion import getCriterion
from data import getDataLoader
from data.dataset import getDataset
from model import getModel
from optim import getOptimizer, getScheduler
from trainer import getTrainer
from util import fixSeed


def main(args, args_file_path, train_log_file_path):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_data_dict = getDataset(args).train_data
    train_data_loader = getDataLoader(
            batch_size=args.TRAIN.BATCH_SIZE,
            dataset=train_data_dict['dataset'],
            sampler=train_data_dict['train_sampler']
            )
    val_data_loader = getDataLoader(
            batch_size=args.TRAIN.BATCH_SIZE,
            dataset=train_data_dict['dataset'],
            sampler=train_data_dict['val_sampler']
            )
    model = getModel(args).to(device)
    criterion = getCriterion(args).to(device)
    optimizer = getOptimizer(args, model)
    scheduler = getScheduler(args, optimizer)
    trainer = getTrainer(args)
    trainer.singleTrain(
            args,
            args_file_path=args_file_path,
            criterion=criterion,
            device=device,
            epoch_num=args.TRAIN.EPOCH_NUM,
            experiment_name=args.TRAIN.MODEL_TYPE.lower(),
            train_log_file_path=train_log_file_path,
            model=model,
            optimizer=optimizer, 
            scheduler=scheduler,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            )

if __name__ == '__main__':
    option = parseConsole()
    args = updateArgs(cfg_file=option.cfg_file_path)
    fixSeed(args.SEED)
    main(args=args, args_file_path=option.args_file_path, 
            train_log_file_path=option.train_log_file_path
            )
