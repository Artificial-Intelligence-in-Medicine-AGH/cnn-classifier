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
import argparse


from TrainingManager import TrainingManager


checkpoint_name = None


def parse_run_arguments() -> None:
    global checkpoint_name

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue_training', action="store", default=None)
    args = parser.parse_args()

    checkpoint_name = args.continue_training  


def main():
    trainer = TrainingManager()
    trainer.train(checkpoint_name=checkpoint_name)


if __name__ == "__main__":
    parse_run_arguments()
    main()
