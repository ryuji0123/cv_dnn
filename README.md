# Sample DNN Repository for Computer Vision
[![GitHub license](https://img.shields.io/github/license/ryuji0123/cv_dnn)](https://github.com/ryuji0123/cv_dnn/blob/main/LICENSE)
![GitHub repo size](https://img.shields.io/github/repo-size/ryuji0123/cv_dnn)
![GitHub Repo stars](https://img.shields.io/github/stars/ryuji0123/cv_dnn?style=social)
![GitHub issues](https://img.shields.io/github/issues/ryuji0123/cv_dnn)


## Dependencies
- Driver Version: 430.26
- CUDA Version: 10.2
- Docker version: 19.03.2

## Steps
1. Install
```sh
$ git clone git@github.com:ryuji0123/cv_dnn.git
```

2. Environment Setup

The names of the docker image and container are specified by constants described in docker/env.sh.
These constants can be edited to suit your project.
```sh
$ cd cv_dnn
$ sh docker/build.sh
$ sh docker/run.sh
$ sh docker/exec.sh
```

4. Run Training Steps
```sh
$ sh nohup_train.sh
```
If you want to stop seeing stdout without breaking train steps, you can just press ctrl+c. The above shell uses nohup and run python script in background, you don't pay attention to how to avoid intrupting train steps. See Logger / Trainer module in detail.

5. See Results
You can use MLflow to check the results of your experiment. 
Access http://localhost:5000/ from your browser. If necessary, you can edit docker/env.sh to change the port.
If you want to see train steps and its status with both stdout and mlflow, you can use tmux. We install it in docker image.


## Components
### Models
Define your model.

model/get_model.py
```py
from pytorch_lightning import LightningModule

from models.simple_cnn_model import SimpleCNNModel


def get_model(args, device) -> LightningModule:
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
```

Since we use Pytorch Lightning, models should inherit "LightningModule". 

model/simple_cnn_model.py
```py
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from optim import get_optimizer, get_scheduler


class SimpleCNNModel(LightningModule):
    def __init__(self, args, device, in_channel, out_channel):
        super(SimpleCNNModel, self).__init__()
        self.args = args
        self._device = device
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # forward
        self.conv1 = nn.Conv2d(in_channel, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_channel)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.args, self)
        scheduler = get_scheduler(self.args, optimizer)

        return [optimizer], [scheduler]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        
        outputs = self(inputs)

        loss = self.cross_entropy_loss(outputs, labels)

        self.log('train_loss', loss, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        
        outputs = self(inputs)

        loss = self.cross_entropy_loss(outputs, labels)

        self.log('val_loss', loss, on_epoch=True, logger=True)

        return loss
```

### Dataset
Define your dataset.

data/dataset/get_datatset.py
```py
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
```

### DataLoader
Define your dataloader.

data/get_data_loader.py
```py
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
```

### Optimizer / Scheduler
Define your optimizer /  scheduler.

optim/get_optimizer.py
```py
from torch.optim import Adam, SGD


def get_optimizer(args, model):
    optimizer_type = args.TRAIN.OPTIMIZER_TYPE.lower()

    if optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=args.TRAIN.LR)
    elif optimizer_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.TRAIN.LR, momentum=args.TRAIN.MOMENTUM)
    else:
        raise NotImplementedError()

    return optimizer

```

optim/get_scheduler.py
```py
from  torch.optim.lr_scheduler import StepLR


def get_scheduler(args, optimizer):
    scheduler_type = args.TRAIN.SCHEDULER_TYPE.lower()

    if scheduler_type == 'step_lr':
        scheduler = StepLR(optimizer, args.TRAIN.STEP_SIZE)
    else:
        raise NotImplementedError()

    return scheduler
```

### Configuration
Currently, we support yacs as a configuration library. You can change default params and add your own configs in "config/defaults.py". Note that you cannot add parameters such as "_C.TRAIN.GPUS" in other files. All parameters should be defined in this file.

config/defaults.py
```py
from os.path import join

from yacs.config import CfgNode

from config.const import PROJECT_ROOT


_C = CfgNode()
_C.SEED = 42


# train
_C.TRAIN = CfgNode()
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.DISTRIBUTED_BACKEND = 'dp'
_C.TRAIN.GPUS = 4
_C.TRAIN.LR = 0.01
_C.TRAIN.MAX_EPOCHS = 2
_C.TRAIN.MODEL_TYPE = 'simple_cnn'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER_TYPE = 'sgd'
_C.TRAIN.SCHEDULER_TYPE = 'step_lr'
_C.TRAIN.STEP_SIZE = 5

_C.MLFLOW = CfgNode()
_C.MLFLOW.EXPERIMENT_NAME = 'Default'


_C.DATA = CfgNode()
_C.DATA.CACHE_DIR = join(PROJECT_ROOT, '.data')
_C.DATA.CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
)
_C.DATA.DATASET_TYPE = 'cifer10'
_C.DATA.INPUT_DIM = 3
_C.DATA.NUM_WORKERS = 32
_C.DATA.TRANSFORM_LIST = ['to_tensor', 'normalize']
_C.DATA.VALIDATION_SIZE = 0.25
```

If you want to change parameters in each model or experiment, you can add your own yaml file. It updates specified parameters in itself. You can specify a yaml file with the command line argument. 

config/simple_cnn.yaml
```yaml
MLFLOW:
  EXPERIMENT_NAME: 'simple_cnn'
```

Also, you can add constant values in "config/const.py"

config/const.py
```py
from os import sep
from os.path import join, dirname, realpath

PROJECT_ROOT = join(sep, *dirname(realpath(__file__)).split(sep)[: -1])
```

### Logger / Trainer
Currently we support MLFlow as an experiment manager and use PyTorch Lightning as a trainer.
Also, we offer logging stdout and saving config as mlflow's artifacts.

train.py
```py
import pytorch_lightning as pl
import pytorch_lightning as pl
import torch

from contextlib import redirect_stdout
from shutil import rmtree

from mlflow.tracking.client import MlflowClient
from pytorch_lightning.loggers import MLFlowLogger

from config import updateArgs, parseConsole
from data import get_dataloader
from data.dataset import get_dataset
from models import get_model


def main(
    args,
    args_file_path: str,
    tmp_results_dir: str,
    train_log_file_path: str,
) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_dict = get_dataset(args).train_data

    train_data_loader = get_dataloader(
        batch_size=args.TRAIN.BATCH_SIZE,
        dataset=train_data_dict['dataset'],
        num_workers=args.DATA.NUM_WORKERS,
        sampler=train_data_dict['train_sampler']
    )

    val_data_loader = get_dataloader(
        batch_size=args.TRAIN.BATCH_SIZE,
        dataset=train_data_dict['dataset'],
        num_workers=args.DATA.NUM_WORKERS,
        sampler=train_data_dict['val_sampler']
    )

    model = get_model(
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
        rmtree(tmp_results_dir, ignore_errors=True)


if __name__ == '__main__':
    option = parseConsole()
    args = updateArgs(cfg_file=option.cfg_file_path)
    main(
        args=args,
        args_file_path=option.args_file_path,
        tmp_results_dir=option.tmp_results_dir,
        train_log_file_path=option.train_log_file_path
    )
```

If you use above train-pipeline, please use "nohup_train.sh" or other similar shell scripts. "nohup_train.sh" uses nohup and some tips to run training steps in background. Once "train.py" started,  you can see stdout as usual and stop it without breaking train steps by just pressing ctrl+c.

nohup_train.sh
```sh
#!/bin/sh
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
TMP_RESULTS_DIR="$(pwd)/.tmp_results/${TIMESTAMP}"
ARGS_FILE="${TMP_RESULTS_DIR}/args.yaml"
TRAIN_LOG_FILE="${TMP_RESULTS_DIR}/log.txt"
mkdir -p $TMP_RESULTS_DIR

export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -u train.py \
  --args_file_path $ARGS_FILE \
  --tmp_results_dir $TMP_RESULTS_DIR \
  --train_log_file_path $TRAIN_LOG_FILE \
  >> $TRAIN_LOG_FILE &
sleep 1s
tail -f $TRAIN_LOG_FILE

```

After training process started, it can be seen on mlflow's application. You can use existing shell script:

```sh
$ sh app_ml.sh
```

Then, please check  localhost:6008 on your web browser.

### Docker
There are three steps to create your own docker environment.
- Build docker image. You can use existing shell script:
```sh
$ sh docker/build.sh
```
Since default user in docker container is root user, user_id and group_id in docker are different from them in host OS. This causes permission problems if you generate files in docker container. To fix this problem, we create duser (docker user). It has the same user_id and group_id with them in host OS, so if you write or edit files, you can access the same files in host OS. Also, duer can use sodo command in docker container without a password, so you don't have to pay attention to settings when you want to install libraries.

- Run docker container. You can use existing shell script:
```sh
$ sh docker/run.sh
```

- Exec docker container. You can use existing shell script:
```sh
$ sh docker/exec.sh
```

### Main
Here's where you combine all previous part.

1. Parse console arguments and obtain configurations as "args".
1. Run "main" function.
1. Create a mlflow session.
1. Create an instance of "Model", "Dataset", "DataLoader"
1. Create an instance of "Trainer" and pass "Model" and "DataLoader" to it.
1. Now you can train your model by calling "trainer.fit()"
