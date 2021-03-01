# Sample DNN Repository for Computer Vision
## Dependency
- Driver Version: 430.26
- CUDA Version: 10.2
- Docker version: 19.03.2

## Steps
1. Install
```sh
$ git clone git@github.com:ryuji0123/cv_dnn.git
$ cd cv_dnn
$ ./build_docker_image.sh
```

2. Environment Setup
```sh
$ ./build_docker_image.sh
$ ./run_docker_container.sh
$ ./exec_docker_container.sh
```

4. Run Training Steps
```sh
$ ./nohup_train.sh
```
If you want to stop seeing stdout without breaking train steps, you can just press ctrl+c. The above shell uses nohup and run python script in background, you don't pay attention to how to avoid intrupting train steps. See Logger & Trainer module in detail.

5. See Results
```sh
$ ./app_ml.sh
```
If you want to see train steps and its status with both stdout and mlflow, you can use tmux. We install it in docker image.


## Components
### Models
Define your own models and add keys. Also, if you want to pass model-dependent hyper parameters, you can add another if-else statement for each model.

model/get_model.py
```
# add models
from model.simple_cnn_model import SimpleCNNModel 


def getModel(args, device):
    # add keys
    model_set = set([
        'simple_cnn'
        ])

    model_type = args.TRAIN.MODEL_TYPE.lower()
    if model_type not in model_set:
        raise NotImplementedError()
    
    # add initializations
    if model_type == 'simple_cnn':
        model = SimpleCNNModel(
                args=args,
                device=device,
                in_channel=args.DATA.INPUT_DIM,
                out_channel=len(args.DATA.CLASSES),
                )

    return model
```

Since we use Pytorch Lightning, models should inherit "LightningModule". 

model/simple_cnn_model.py
```
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from pytorch_lightning import LightningModule

from optim import getOptimizer, getScheduler


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
        optimizer = getOptimizer(self.args, self)
        scheduler = getScheduler(self.args, optimizer)

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
Define your dataset and add keys.

data/dataset/get_datatset.py
```
# add datasets
from data.dataset.cifer_10_dataset import CIFAR10Dataset 


def getDataset(args):
    # add keys
    dataset_ref = set(['cifer10']) 

    dataset_type = args.DATA.DATASET_TYPE.lower()
    if dataset_type not in dataset_ref:
        raise NotImplementedError()

    transform = getTransformFromList(args.DATA.TRANSFORM_LIST)

    if dataset_type == 'cifer10':
        dataset = CIFAR10Dataset(
            root=args.DATA.CACHE_DIR, transform=transform, val_size=args.DATA.VAL_SIZE
            )

    return dataset
```

### DataLoader
Define your own dataloader and add keys.

data/get_data_loader.py
```
# add data loader
from torch.utils.data import DataLoader 


def getDataLoader(batch_size, dataset, sampler=None):
    # you can add another data loader using if-else statement
    return DataLoader( 
            batch_size=batch_size, dataset=dataset, sampler=sampler,
            shuffle=False if sampler is not None else True
            )
```

### Optimizer / Scheduler
Define your own optimizer /  scheduler and add keys.

optim/get_optimizer.py
```
from torch.optim import Adam, SGD # add optimizers


def getOptimizer(args, model):
    optimizer_set = set(['adam', 'sgd']) # add keys
    optimizer_type = args.TRAIN.OPTIMIZER_TYPE.lower()
    if optimizer_type not in optimizer_set:
        raise NotImplementedError()

    if optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=args.TRAIN.LR)
    elif optimizer_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.TRAIN.LR, momentum=args.TRAIN.MOMENTUM)

    return optimizer
```

optim/get_scheduler.py
```
from  torch.optim.lr_scheduler import StepLR # add schedulers


def getScheduler(args, optimizer):
    scheduler_set = set(['step_lr']) # add keys
    scheduler_type = args.TRAIN.SCHEDULER_TYPE.lower()
    if scheduler_type not in scheduler_set:
        raise NotImplementedError()

    if scheduler_type == 'step_lr':
        scheduler = StepLR(optimizer, args.TRAIN.STEP_SIZE)

    return scheduler
```

### Configuration
Currently, we support yacs as a configuration library. You can change default params and add your own configs in "config/defaults.py". Note that you cannot add parameters such as "_C.TRAIN.GPUS" in other files. All parameters should be defined in this file.

config/defaults.py
```
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
_C.DATA.VAL_SIZE = 0.25
```

If you want to change parameters in each model or experiment, you can add your own yaml file. It updates specified parameters in itself. You can specify a yaml file with the command line argument. 

config/simple_cnn.yaml
```
MLFLOW:
  EXPERIMENT_NAME: 'simple_cnn'
```

Also, you can add constant values in "config/const.py"

config/const.py
```
from os import sep
from os.path import join, dirname, realpath

PROJECT_ROOT = join(sep, *dirname(realpath(__file__)).split(sep)[: -1])
```

### Logger & Trainer
Currently we support MLFlow as an experiment manager and use PyTorch Lightning as a trainer.
Also, we offer logging stdout and saving config as mlflow's artifacts.

train.py
```
from contextlib import redirect_stdout
from shutil import rmtree

from mlflow.tracking.client import MlflowClient
from pytorch_lightning.loggers import MLFlowLogger

def main(args, args_file_path, tmp_results_dir, train_log_file_path):
=============================================================================
=============================================================================
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
```

If you use above train-pipeline, please use "nohup_train.sh" or other similar shell scripts. "nohup_train.sh" uses nohup and some tips to run training steps in background. Once "train.py" started,  you can see stdout as usual and stop it without breaking train steps by just pressing ctrl+c.

nohup_train.sh
```
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

```cv_dnn/
$ ./app_ml.sh
```

Then, please check  localhost:6008 on your web browser.

### Docker
There are three steps to create your own docker environment.
- Build docker image. You can use existing shell script:
```cv_dnn/
$ ./build_docker_image.sh
```
Since default user in docker container is root user, user_id and group_id in docker are different from them in host OS. This causes permission problems if you generate files in docker container. To fix this problem, we create duser (docker user). It has the same user_id and group_id with them in host OS, so if you write or edit files, you can access the same files in host OS. Also, duer can use sodo command in docker container without a password, so you don't have to pay attention to settings when you want to install libraries.

- Run docker container. You can use existing shell script:
```cv_dnn/
$ ./run_docker_container.sh
```

- Exec docker container. You can use existing shell script:
```cv_dnn/
$ ./exec_docker_container.sh
```

### Main
Here's where you combine all previous part.

1. Parse console arguments and obtain configurations as "args".
1. Run "main" function.
1. Create a mlflow session.
1. Create an instance of "Model", "Dataset", "DataLoader"
1. Create an instance of "Trainer" and pass "Model" and "DataLoader" to it.
1. Now you can train your model by calling "trainer.fit()"
