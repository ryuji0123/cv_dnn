import mlflow
import torch

from contextlib import redirect_stdout
from os import remove

from tqdm import tqdm

from trainer.util import getMlflowExperimentIdByName


class ParentTrainer:

    def __init__(self):
        pass

    def acc(self, outputs, labels):
        return (outputs.argmax(1) == labels).sum().item()

    def singleTrain(
            self, args, args_file_path, criterion, device, epoch_num, experiment_name, model,
            optimizer, scheduler, train_data_loader, train_log_file_path, val_data_loader,
            ):

        with open(args_file_path, 'w') as f:
            with redirect_stdout(f):
                print(args)

        with mlflow.start_run(experiment_id=getMlflowExperimentIdByName(experiment_name)):
            mlflow.log_artifact(args_file_path)
            remove(args_file_path)

            best_val_acc = 0.0

            for epoch in range(epoch_num):
                train_loss, train_acc = self.train(
                        criterion=criterion,
                        device=device,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        train_data_loader=train_data_loader
                        )

                val_loss, val_acc = self.val(
                        criterion=criterion,
                        device=device,
                        model=model,
                        val_data_loader=val_data_loader
                        )

                mlflow.log_metric(key='train_loss', value=train_loss, step=epoch)
                mlflow.log_metric(key='val_loss', value=val_loss, step=epoch)
                mlflow.log_metric(key='train_acc', value=train_acc, step=epoch)
                mlflow.log_metric(key='val_acc', value=val_acc, step=epoch)

                if best_val_acc < val_acc:
                    mlflow.pytorch.log_model(
                            pytorch_model=model, artifact_path=f'model_epoch={epoch}'
                            )
                    best_val_acc = val_acc
            
            mlflow.log_artifact(train_log_file_path)
            remove(train_log_file_path)

    def train(self, criterion, device, model, optimizer, scheduler, train_data_loader):
        train_loss = train_acc = 0.0
        model.train()

        for inputs, labels in tqdm(train_data_loader):
            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
                
            loss = criterion(outputs, labels) 

            train_acc += self.acc(outputs, labels)
            
            train_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()
            
        scheduler.step()

        length = len(train_data_loader.dataset)
        return [x / length for x in [train_loss, train_acc]]

    def val(self, criterion, device, model, val_data_loader): 
        val_loss = val_acc = 0.0
        model.eval()

        with torch.no_grad():
            for inputs, labels in tqdm(val_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device) 

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += self.acc(outputs, labels)
        
        length = len(val_data_loader.dataset)
        return [x / length for x in [val_loss, val_acc]]
