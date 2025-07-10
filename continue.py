import os

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

from config import config
from dataset_manager.dataset_manager import get_loaders, REF_CLASSES
from helper_scripts.logs_plots import save_logs_as_plots
from training_manager import training_manager


import sys
import time


if len(sys.argv) < 2:
    print("Please provide a name of saved model you wish to train further")

hyperparameters = config.hyperparameters
LOGS_PATH = config.logs_path

def main():
    log_file = open(f"{LOGS_PATH}/training_continiuation.log","w")
    sys.stdout = log_file

    train = training_manager()

    #################################
    n_epoch = hyperparameters.total_epoch
    n_save = hyperparameters.save_every
    ################################

    print(f"Running on device: {train.device} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    print(f"============================ MODEL {config.model_name} FROM {sys.argv[1]} FILE ============================")


    train_loader, val_loader = get_loaders()


    checkpoint = train.load_chceckpoint(sys.argv[1])

    last_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_auc = checkpoint['best_auc']
    logs = checkpoint['logs']


    for epoch in range(last_epoch+1,n_epoch):
        start = time.time()

        print(f"\n================\nEpoch {epoch + 1}")
        
        train_loss = train.training_step(train_loader=train_loader)
        
        print(f"\nTrain Loss: {train_loss:.4f}")

        val_preds, val_labels, val_loss = train.validation_step(val_loader=val_loader)

        print(f"Validation Loss: {val_loss:.4f}")

        val_auc = roc_auc_score(
            y_true=val_labels,
            y_score=val_preds,
            multi_class="ovr",
        )
        train.scheduler.step(val_auc)

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

        logs["epoch"].append(epoch)
        logs["epoch_time"].append(epoch_time)
        logs["train_loss"].append(train_loss)
        logs["val_loss"].append(val_loss)
        logs["val_auc"].append(val_auc)
        logs["val_accuracy"].append(val_accuracy)



        if val_loss < best_val_loss:
            best_val_loss = val_loss
            train.save_checkpoint("Best_Loss", epoch, best_val_loss, best_auc, logs)
            print(f"Best loss model saved")

        if val_auc > best_auc:
            best_auc = val_auc
            train.save_checkpoint("Best_AUC", epoch, best_val_loss, best_auc, logs)
            print(f"Best auc model saved")

        if epoch % n_save == 0:
            save_logs_as_plots(logs=logs, save_path=LOGS_PATH)
            train.save_checkpoint("Latest", epoch, best_val_loss, best_auc, logs)
            print(f"Latest model saved")



    train.save_checkpoint("Final", config.hyperparameters.total_epoch, best_val_loss, best_auc, logs)
    print(f"Final model saved")

    log_file.close()
    print("Training completed.")


if __name__ == "__main__":
    main()
