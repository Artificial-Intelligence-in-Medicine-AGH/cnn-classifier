# cardiomegaly cnn team2

Multiclassification algorithm created using Convolutional Neural Network

## First time setup

Use python3.11, create virtual environment and install required packages

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
    "dataset_path": "path/to/folder",
    "labels_file_path": "path/to/file",
    "logs_path": "path/to/folder",
    "save_model_path": "path/to/folder",
    "model_name": "tf_efficientnet_b0.in1k",
    "final_img_width": 224,
    "num_channels": 1,
    "hyperparameters":{
        "learning_rate": 0.001,
        "batch_size": 16,
        "total_epoch": 100,
        "save_every":10,
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
    }
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
