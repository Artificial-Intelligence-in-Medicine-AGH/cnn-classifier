#TODO imports
import timm
import torch
import torch.nn as nn


from config import config


#TODO create predictor class
class Predictor():
    def __init__(self, model_name:str):
        #TODO: Load Model models_path + model_name for predicting
        pass
    
    def _predict(self, img:torch.tensor):
        #TODO: Implement method for predicting classes from given image (tensor)
        # the main idea is keeping math behind model prediction in seperate private method
        pass
        
    def predict_from_dir(self, dir_path:str):
        #TODO: Create script that would call _predict for photos in given directory and them organize data in somekind of csv or plot 
        pass
    
