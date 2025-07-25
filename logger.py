from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt

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

        self.current_log = ""
        self.log_file_name = f"{config.logs_path}/training_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}.log"

    def __call__(self, line:str) -> str:
        with open(self.log_file_name, "a") as f:
            print(line, file = f)
        
        return line

    def save_params(self, epoch:int, epoch_time:float, train_loss:float, val_loss:float,val_auc:float, val_accuracy:float) -> None:
        self.logs["epoch"].append(epoch)
        self.logs["epoch_time"].append(epoch_time)
        self.logs["train_loss"].append(train_loss)
        self.logs["val_loss"].append(val_loss)
        self.logs["val_auc"].append(val_auc)
        self.logs["val_accuracy"].append(val_accuracy)
    
    def set_logs(self, logs:dict[str]) -> dict:
        self.logs = logs
        return logs

    def get_logs(self) -> dict:
        return self.logs
    
    def plot(self) -> str:
        save_path = os.path.join(config.logs_path, 'plots')
        os.makedirs(save_path, exist_ok=True)
        
        metrics = [key for key in self.logs if key != "epoch"]

        for metric in metrics:
            plt.figure(figsize=(8, 6))
            plt.plot([x+1 for x in self.logs["epoch"]], self.logs[metric], marker='o', label=metric)
            plt.title(f"{metric.replace('_', ' ').title()} Over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.xlim([0, max(self.logs["epoch"]) + 5])
            plt.ylim([0, 1.1*max(self.logs[metric])])

            filename = f"{metric}.png"
            plt.savefig(os.path.join(save_path, filename))
            plt.close()

        return save_path