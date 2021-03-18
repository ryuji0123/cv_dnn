import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from pytorch_lightning import LightningModule

from optim import get_optimizer, get_scheduler


class SimpleCNNModel(LightningModule):
    def __init__(
        self,
        args,
        device: str,
        hparams: dict,
        in_channel: int,
        out_channel: int,
        trial,
    ) -> None:
        super(SimpleCNNModel, self).__init__()
        self.args = args
        self._device = device
        self.hparams = hparams
        self.trial = trial

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # forward
        self.conv1 = nn.Conv2d(in_channel, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_channel)

    def configure_optimizers(self) -> list:
        if self.trial is not None:
            lr = self.trial.suggest_loguniform('optimizer_lr', 1e-5, 1e-1)
            self.logger.log_hyperparams({'optimizer_lr': lr})
        else:
            lr = self.args.TRAIN.LR
        optimizer = get_optimizer(self.args, self, lr=lr)
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

        accuracy = (outputs.argmax(1) == labels).sum().item()
        loss = self.cross_entropy_loss(outputs, labels)

        return OrderedDict({
            'accuracy': accuracy,
            'count': labels.shape[0],
            'loss': loss,
        })

    def training_epoch_end(self, outputs) -> None:
        accuracy = loss = 0.0
        count = 0
        for output in outputs:
            accuracy += output['accuracy']
            loss += output['loss'].data.item()
            count += output['count']

        results = {
            'training_accuracy': accuracy / count,
            'training_loss': loss / count,
        }

        self.logger.log_metrics(results, step=self.current_epoch)

        return None

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        outputs = self(inputs)

        accuracy = (outputs.argmax(1) == labels).sum().item()
        loss = self.cross_entropy_loss(outputs, labels)

        return OrderedDict({
            'accuracy': accuracy,
            'count': labels.shape[0],
            'loss': loss,
        })

    def validation_epoch_end(self, outputs) -> None:
        accuracy = loss = 0.0
        count = 0
        for output in outputs:
            accuracy += output['accuracy']
            loss += output['loss'].data.item()
            count += output['count']

        results = {
            'validation_accuracy': accuracy / count,
            'validation_loss': loss / count,
        }

        self.logger.log_metrics(results, step=self.current_epoch)

        return results
