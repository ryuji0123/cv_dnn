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
