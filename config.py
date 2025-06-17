import os

from pydantic import BaseModel
from typing import Optional, Dict
import json

from typing import Literal

class SchedulerParams(BaseModel):
    mode: Literal["min", "max"]
    factor: float
    patience: int


class Hyperparameters(BaseModel):
    learning_rate:float
    batch_size: int
    total_epoch:int
    save_every:int

    weight_decay:float
    max_norm:int
    drop_rate:float
    num_blocks_to_unfreeze:int

    scheduler_type: str
    scheduler_params: SchedulerParams

class Config(BaseModel):
    # PATHS
    dataset_path:str
    labels_file_path:str
    logs_path:str
    save_model_path:str

    # model
    model_name:str
    
    # Image  
    final_img_width:int
    num_channels: int

    hyperparameters:Hyperparameters

    @staticmethod 
    def load_from_file(file_path:str):
        with open(file_path, "r") as f:
            file_content_str = f.read()
            file_dict = json.loads(file_content_str)
            conf = Config(**file_dict)
            return conf

# Needed in case the script was run from different directory than the repo root
repo_root_folder = os.path.dirname(os.path.abspath(__file__))

# This can be imported in other files
config = Config.load_from_file(os.path.join(repo_root_folder, "config.json"))
