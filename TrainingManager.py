import timm
import torch
import torch.nn as nn

import numpy as np
import os


from config import config
from dataset_manager.dataset_manager import get_loaders, REF_CLASSES


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
        

    
    def training_step(self, train_loader:torch.utils.data.DataLoader) -> float:
        
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
    

    def validation_step(self, val_loader:torch.utils.data.DataLoader) -> (np.ndarray, np.ndarray, float):  # type: ignore
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
    

    def save_checkpoint(self, name:str, epoch:int, best_val_loss:float, best_auc:float, logs:dict) -> dict:
        checkpoint = {
            'epoch':epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_auc': best_auc,
            'logs': logs
        }
        model_path = os.path.join(config.save_model_path, f"{name}.pth")
        torch.save(checkpoint, model_path)

        return checkpoint
    
    def load_chceckpoint(self, name:str) -> dict:
        checkpoint = torch.load(os.path.join(config.save_model_path, name))
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
    
        return checkpoint
    