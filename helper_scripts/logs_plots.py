import os
import matplotlib.pyplot as plt

def save_logs_as_plots(logs: dict, save_path: str):
    os.makedirs(save_path, exist_ok=True)

    metrics = [key for key in logs if key != "epoch"]

    for metric in metrics:
        plt.figure(figsize=(8, 6))
        plt.plot(logs["epoch"], logs[metric], marker='o', label=metric)
        plt.title(f"{metric.replace('_', ' ').title()} Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.xlim([0, max(logs["epoch"]) + 5])
        plt.ylim([0, 1.1*max(logs[metric])])

        filename = f"{metric}.png"
        plt.savefig(os.path.join(save_path, filename))
        plt.close()