# cnn classifier

Multiclassification algorithm created using Convolutional Neural Network

## First time setup

Use python3.11, create virtual environment and install required packages

Install pytorch by your preferences with official [PyTroch Get Started guide](https://pytorch.org/get-started/locally/)

NOTE for AMD gpu users: try installing pytorch using following command BEFORE installing all other requirements:
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

Installing other requirements:
```shell
pip install -r requirements.txt
```

## Configuration File
Note that conf.json should be mannualy created in main directory (the same that contains config.py)

Here is an example of config.json:
```json
{
    "dataset_path": "/home/franio/Desktop/cnn-classifier/data",
    "labels_file_path": "/home/franio/Desktop/cnn-classifier/data/small_data_entry.csv",
    "logs_path": "/home/franio/Desktop/cnn-classifier/logs",
    "save_model_path": "/home/franio/Desktop/cnn-classifier/model",
    "model_name": "tf_efficientnet_b0.in1k",
    "final_img_width": 224,
    "num_channels": 1,
    "hyperparameters":{
        "learning_rate": 0.001,
        "batch_size": 1,
        "total_epoch": 100,
        "save_every":1,
        "weight_decay":0.0001,
        "max_norm":1,
        "drop_rate":0.3,
        "num_blocks_to_unfreeze": 2,
        "scheduler_type": "ReduceLROnPlateau",
        "scheduler_params": {
            "mode": "max",
            "factor": 0.1,
            "patience": 2
        }
    },

    "classes":  [
    "Cardiomegaly",
    "Emphysema",
    "Effusion",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Atelectasis",
    "Pneumothorax",
    "Pleural_Thickening",
    "Pneumonia",
    "Fibrosis",
    "Edema",
    "Consolidation"
    ],

    "no_class_name":"No Finding" //WIP: Moving classes to bit maps, rethink preprocessing
}
```

## Preprocessing
Preprocessing resizes all images, changes their format to `.pth` and split them into train, val, test subfolders of `config.dataset_path` 

To run preprocessing call preprocessing script with  directory containing data as an program argument:
```cmd
preprocessing.py <data_dir>
```
Note that preprocessing script copy date instead of moving it.

### Directory structures
Data will be splitted into val, train and test directories in the following directory structure:
<br/>/`config.dataset_path`
<br/>├──/train
<br/>├──/val
<br/>└──/test

Each of them will be created in one directory with path to it provided in configuration file.
