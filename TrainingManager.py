import timm
import torch
import torch.nn as nn

import numpy as np
import os
import time
import sys

from config import config
from dataset_manager.dataset_manager import get_loaders, REF_CLASSES
from helper_scripts.logs_plots import save_logs_as_plots

from sklearn.metrics import roc_auc_score, accuracy_score

hyperparameters = config.hyperparameters

class TrainingManager():
    def __init__(self):
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = timm.create_model(config.model_name, pretrained=True, in_chans=config.num_channels, num_classes=len(REF_CLASSES),
                              drop_rate=hyperparameters.drop_rate).to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # Unfreeze the last N blocks
        for block in self.model.blocks[-hyperparameters.num_blocks_to_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                 lr=hyperparameters.learning_rate, weight_decay=hyperparameters.weight_decay)

        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=hyperparameters.scheduler_params.mode,
                patience=hyperparameters.scheduler_params.patience,
                factor=hyperparameters.scheduler_params.factor,
            )
        
        self.epoch = 0
        self.best_auc = 0
        self.best_val_loss = 1e10
        self.logs = {
            "epoch": [],
            "epoch_time": [],
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_accuracy": [],
        }

        self.train_loader, self.val_loader  = get_loaders()


    def __training_step(self, train_loader:torch.utils.data.DataLoader) -> float:
        
        self.model.train()
        train_loss = 0.0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=hyperparameters.max_norm, norm_type=2)
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_loss += loss.item()

        return train_loss / len(train_loader)
    

    def __validation_step(self, val_loader:torch.utils.data.DataLoader) -> (np.ndarray, np.ndarray, float):  # type: ignore
        self.model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                val_preds.append(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_labels = np.array(val_labels)
        val_loss = val_loss / len(val_loader)

        return val_preds, val_labels, val_loss
    

    def __save_checkpoint(self, name:str) -> dict:
        checkpoint = {
            'epoch':self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_auc': self.best_auc,
            'logs': self.logs
        }
        model_path = os.path.join(config.save_model_path, f"{name}.pth")
        torch.save(checkpoint, model_path)

        return checkpoint
    
    def load_chceckpoint(self, name:str) -> dict:
        checkpoint = torch.load(os.path.join(config.save_model_path, name))
        
        self.epoch = checkpoint['epoch']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.best_val_loss = checkpoint['best_val_loss']
        self.best_auc = checkpoint['best_auc']
        self.logs = checkpoint['logs']
    
        return checkpoint
    

    def train(self, checkpoint_name:str):   
        
        if checkpoint_name is not None:
            self.load_chceckpoint(checkpoint_name)
            log_file = open(f"{config.logs_path}/training.log","a")
            sys.stdout = log_file

            print(f"\n\nReasuming training from {checkpoint_name} file")
            print(f"Running on device: {self.device} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
            print(f"============================ MODEL {config.model_name} ============================")

        else:
            log_file = open(f"{config.logs_path}/training.log","w")
            sys.stdout  = log_file
            
            print(f"Running on device: {self.device} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
            print(f"============================ MODEL {config.model_name} ============================")

        #################################
        n_save = hyperparameters.save_every
        n_epoch = hyperparameters.total_epoch
        #################################

        for epoch in range(self.epoch, n_epoch):
            start = time.time()
            print(f"\n================\nEpoch {epoch + 1}")
            
            train_loss = self.__training_step(train_loader=self.train_loader)
            
            print(f"\nTrain Loss: {train_loss:.4f}")

            val_preds, val_labels, val_loss = self.__validation_step(val_loader=self.val_loader)

            print(f"Validation Loss: {val_loss:.4f}")

            val_auc = roc_auc_score(
                y_true=val_labels,
                y_score=val_preds,
                multi_class="ovr",
            )
            self.scheduler.step(val_auc)

            # Needed by accuracy_score classifier
            THRESHOLD = 0.5
            val_preds_binary = (val_preds > THRESHOLD)
            val_accuracy = accuracy_score(
                y_true=val_labels,
                y_pred=val_preds_binary,
            )

            stop = time.time()
            epoch_time = stop - start

            print(f"Validation auc score: {val_auc}")
            print(f"Validation accuracy score: {val_accuracy}")
            print(f"Epoch time: {round(epoch_time,2)} s")

            self.logs["epoch"].append(epoch)
            self.logs["epoch_time"].append(epoch_time)
            self.logs["train_loss"].append(train_loss)
            self.logs["val_loss"].append(val_loss)
            self.logs["val_auc"].append(val_auc)
            self.logs["val_accuracy"].append(val_accuracy)


            self.epoch = epoch + 1

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.__save_checkpoint("Best_Loss")
                print(f"Best loss model saved")

            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.__save_checkpoint("Best_auc")
                print(f"Best auc model saved")

            if epoch % n_save == 0:
                save_logs_as_plots(logs=self.logs, save_path=config.logs_path)
                self.__save_checkpoint("Latest")
                print(f"Latest model saved")

            


        self.__save_checkpoint("Final", config.hyperparameters.total_epoch, self.best_val_loss, self.best_auc, self.logs)
        print(f"Final model saved")

        log_file.close()
        print("Training completed.")