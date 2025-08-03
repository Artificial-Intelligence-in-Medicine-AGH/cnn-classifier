import os

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

from config import config
from dataset_manager.dataset_manager import get_loaders, REF_CLASSES
from logger import Logger

import sys
import time


#TODO implement predicting from folder
