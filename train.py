import os

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

from config import config
from dataset_manager.dataset_manager import get_loaders, REF_CLASSES
from helper_scripts.logs_plots import save_logs_as_plots

import sys
import time

hyperparameters = config.hyperparameters
scheduler_params = hyperparameters.scheduler_params

TRAIN_DATA_PATH = os.path.join(config.dataset_path, "train")
VAL_DATA_PATH = os.path.join(config.dataset_path, "val")
TEST_DATA_PATH = os.path.join(config.dataset_path, "test")
LOGS_PATH = config.logs_path
SAVE_MODEL_PATH = config.save_model_path


def main():
    log_file = open(f"{LOGS_PATH}/training.log","w")
    sys.stdout = log_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    #################################
    weight_decay = hyperparameters.weight_decay
    max_norm = hyperparameters.max_norm
    drop_rate = hyperparameters.drop_rate
    n_epoch = hyperparameters.total_epoch
    n_save = hyperparameters.save_every
    ################################

    train_loader, val_loader = get_loaders()

    num_classes = len(REF_CLASSES)

    model = timm.create_model(config.model_name, pretrained=True, in_chans=config.num_channels, num_classes=num_classes,
                              drop_rate=drop_rate).to(device)

    print(f"============================ MODEL {config.model_name} ============================")

    best_auc = 0
    best_val_loss = 1e10

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    # Unfreeze the last N blocks
    for block in model.blocks[-hyperparameters.num_blocks_to_unfreeze:]:
        for param in block.parameters():
            param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=hyperparameters.learning_rate, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_params.mode,
        patience=scheduler_params.patience,
        factor=scheduler_params.factor,
    )

    logs = {
        "epoch": [],
        "epoch_time": [],
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_accuracy": [],
    }
    for epoch in range(n_epoch):
        start = time.time()
        print(f"\n================\nEpoch {epoch + 1}")
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=2)
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        print(f"\nTrain Loss: {train_loss:.4f}")

        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                val_preds.append(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_labels = np.array(val_labels)
        val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        val_auc = roc_auc_score(
            y_true=val_labels,
            y_score=val_preds,
            multi_class="ovr",
        )
        scheduler.step(val_auc)

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
            model_path = os.path.join(SAVE_MODEL_PATH, f"Best_Loss.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Best loss model saved")

        if val_auc > best_auc:
            best_auc = val_auc
            model_path = os.path.join(SAVE_MODEL_PATH, f"Best_AUC.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Best auc model saved")

        if epoch % n_save == 0:
            save_logs_as_plots(logs=logs, save_path=LOGS_PATH)
            model_path = os.path.join(SAVE_MODEL_PATH, f"Latest.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Latest model saved")

    model_path = os.path.join(SAVE_MODEL_PATH, f"Final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved")

    log_file.close()
    print("Training completed.")


if __name__ == "__main__":
    main()
