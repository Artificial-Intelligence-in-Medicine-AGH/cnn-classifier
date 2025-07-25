from datetime import datetime
import sys


from config import config


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

        self.log_file = open(f"{config.logs_path}/training_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}.log","w") 

    def __call__(self, line:str):
        print(line, file = self.log_file)

    def save_params(self, epoch, epoch_time, train_loss, val_loss,val_auc, val_accuracy ):
        self.logs["epoch"].append(epoch)
        self.logs["epoch_time"].append(epoch_time)
        self.logs["train_loss"].append(train_loss)
        self.logs["val_loss"].append(val_loss)
        self.logs["val_auc"].append(val_auc)
        self.logs["val_accuracy"].append(val_accuracy)
    
    
    def __del__(self):
        self.log_file.close()