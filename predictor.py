# TODO imports
import timm
import torch
import torch.nn as nn
import numpy as np
import os

from config import config

# TODO: import _load_model from the other file once created
# from model_loader import _load_model


class Predictor(nn.Module):
    def __init__(self, models_path:str, model_name: str):
        super().__init__()
        self.device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
        self.models_path: str = models_path
        self.name: str = model_name
        self.model_path = os.path.join(models_path, model_name)


        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found {self.model_path}.")
    
        # Calling the external _load_model function
        _load_model(self)

        self.to(self.device)

        print(f"Predictor initialized with model: {self.model_path} on device: {self.device}")

    @torch.no_grad()
    def _predict(self, img: torch.Tensor):
        """
        Run inference on a single preprocessed image tensor and return predicted label and confidence.

        Parameters
        ----------
        img : torch.Tensor
            Aproprietly preprocessed image.

        Returns
        -------
        dict
            A dictionary with keys:
            - 'label' (str): predicted class name from config.classes
            - 'confidence' (float): probability assigned to the predicted class in [0.0, 1.0]

        Notes
        -----
        - The method sets the model to eval() and runs inference with gradients disabled.
        - The input tensor is moved to the predictor's device (self.device) before forwarding.
        - Softmax is applied to the model output along the last dimension; results are converted to a NumPy array.
        - Ensure the model is loaded on the predictor and config.classes matches the model's output dimension.
        """
        
        self.eval()
        # move Tensor to active device
        img: torch.Tensor = img.to(self.device)

        # self(img).detach().cpu() -> raw predictions from model placed on cpu
        # apply softmax to predictions and turn everthing to numpy ndarray

        # TODO: check output of the model (1d/2d) or make it universal somehow

        probabilities: np.ndarray = nn.functional.softmax(self(img).detach().cpu()).numpy(force=True).flatten()
        
        # return label with the highest score and the score itself
        return {'label': config.classes[np.argmax(probabilities)], 
                'confidence': np.max(probabilities)}

        
    def predict_from_dir(self, dir_path:str):
        #TODO: Create script that would call _predict for photos in given directory and them organize data in somekind of csv or plot 
        pass

