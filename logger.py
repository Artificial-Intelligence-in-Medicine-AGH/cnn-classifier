from datetime import datetime




class Logger():
    def __init__(self):
        self.logs = {
            "epoch": [],
            "epoch_time": [],
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_accuracy": [],
        }
