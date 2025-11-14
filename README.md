# CNN Classifier

A multiclass image classification project built using **Convolutional Neural Networks (CNN)** and **PyTorch**.  
This model aims to classify various heart diseases based on medical images.

---

## Features

- Multi-class classification using deep CNN architecture  
- Modular and configurable training system  
- Automated logging and visualization  
- Configurable preprocessing and dataset management  

---

## First-Time Setup

### 1. Python Environment

Use **Python 3.11**.  
Create a virtual environment and install required packages:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

---

### 2. Install PyTorch

Follow the [official PyTorch Get Started Guide](https://pytorch.org/get-started/locally/)  
to install PyTorch for your system.

> **NOTE for AMD GPU users:**  
> Install PyTorch manually before other requirements using:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
> ```

---

### 3. Install Other Requirements

Once PyTorch is installed, install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration File

Create a configuration file named **`config.json`** in the project root (same directory as `config.py`).

### Example `config.json`

```json
{
    "dataset_path": "/home/user/cnn-classifier/data",
    "labels_file_path": "/home/user/cnn-classifier/data/labels.csv",
    "logs_path": "/home/user/cnn-classifier/logs",
    "save_model_path": "/home/user/cnn-classifier/model",
    "model_name": "tf_efficientnet_b0.in1k",
    "final_img_width": 224,
    "num_channels": 1,
    "hyperparameters": {
        "learning_rate": 0.001,
        "batch_size": 1,
        "total_epoch": 100,
        "save_every": 1,
        "weight_decay": 0.0001,
        "max_norm": 1,
        "drop_rate": 0.3,
        "num_blocks_to_unfreeze": 2,
        "scheduler_type": "ReduceLROnPlateau",
        "scheduler_params": {
            "mode": "max",
            "factor": 0.1,
            "patience": 2
        }
    },
    "classes": [
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
    ]
}
```

---

## Preprocessing

Before training, you must preprocess your dataset.

Preprocessing performs:

- Image resizing to the target width  
- Format conversion to `.pth` tensors  
- Automatic train/validation/test splitting  

### Run Preprocessing

```bash
python preprocessing.py <data_dir>
```

> **Note:** The preprocessing script **copies** files instead of moving them.

---

## Directory Structure

After preprocessing, your dataset will follow this structure:

```
/config.dataset_path
├── train/
├── val/
└── test/
```

Each directory will be automatically created based on paths provided in the configuration file.

## Logging

Training logs and plots are automatically generated under:

```
/logs/
```

Metrics such as **loss**, **accuracy**, and **AUC** are plotted and saved as `.png` images for each epoch.

---

## License

This project is released under the **MIT License**.  
Feel free to use and modify it for educational and research purposes.
