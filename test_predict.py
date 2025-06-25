import sys

import torch
import torch.nn as nn
import timm
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, InterpolationMode
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import nibabel as nib
from dataset_manager.dataset_manager import CustomDataset, REF_CLASSES, get_test_loader
from config import config

# TODO: import config and needed paths etc..
# TODO: reimplement model pred saving, also update config
# TODO: add our classes and edit other needed dependencies


# TODO: update directories and model name from config
# TODO: get rid of fold mechanisim

if len(sys.argv) < 2:
    print("ERROR: please provide a trained model name as a argument of the script")
    exit()


# TODO: give aliases for model files names
TEST_DIR = os.path.join(config.dataset_path, "test")
MODEL_PATH = os.path.join(config.save_model_path, f"{sys.argv[1]}.pth")
NUM_CLASEES = len(REF_CLASSES)

def main(logs_base_model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    model = timm.create_model(
        config.model_name,
        pretrained=True,
        in_chans=config.num_channels,
        num_classes=NUM_CLASEES,
        drop_rate=config.drop_rate,
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    test_loader = get_test_loader()

    predictions = []
    test_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probabilities = torch.sigmoid(outputs, dim=1).cpu().numpy()
            predictions.append(probabilities)
            test_labels.extend(labels)



    #TODO: REST OF CODE TO REWRITE
    # Aggregate and calculate AUC


    test_auc = roc_auc_score(
        y_true=test_labels
    y_score=ensemble_predictions,
    average="macro",
    multi_class="ovr",
    )

    f = open(logs_base_model + '/names_' + mode + '_.txt','w')
    for item in fnames:
        print(item,file=f)
    f.close()

    f = open(logs_base_model + '/true_' + mode + '_.txt','w')
    for item in test_labels:
        print(item,file=f)
    f.close()

    f = open(logs_base_model + '/predictions_' + mode + '_.txt','w')
    for i in range(ensemble_predictions.shape[0]):
        print(ensemble_predictions[i],file=f)
    f.close()

    f = open(logs_base_model + '/results_' + mode + '_.txt','w')
    print("=== Test Results ===",file=f)
    print(f"Test Set AUC: {test_auc:.4f}",file=f)
    f.close()

if __name__ == "__main__":

    work_dir = sys.argv[1]
    mode = sys.argv[2]
    main(work_dir, mode)