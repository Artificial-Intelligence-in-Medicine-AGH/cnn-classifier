"""
This module implements a Predictor that prepares the model for training.

The model retrival is implemented as abstract factory design pattern for retrival via torch and timm.

Please see https://refactoring.guru/design-patterns/abstract-factory/python/example for reference on how abstract
factory works.
"""

from abc import abstractmethod, ABC
from enum import IntEnum, auto
from pathlib import Path
from typing import override, final

import timm
import torch
import torch.nn as nn
import numpy as np
import os

from timm.models import PretrainedCfg

from config import config


class Predictor:
    def __init__(self, models_path: str, model_name: str) -> None:
        """Object used to make predictions on given image tensor

        Args:
            models_path (str): path to folder with models
            model_name (str): name of model

        Raises:
            FileNotFoundError: when directory `models_path` does not exist
        """
        self.device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
        self.name: str = model_name
        self.model_path = os.path.join(models_path, model_name)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found {self.model_path}.")

        # model loader
        self.model_loader_config: ModelLoaderConfig = ModelLoaderConfig(self.device, models_path, model_name)
        self.loader_factory: ModelLoader = get_loader_factory(loader_type=ModelLoaderType.TIMM_MODEL_FACTORY,
                                                              loader_config=self.model_loader_config)
        
        # loading model
        self._model: torch.nn.Module = self._load_model()
        self._model.to(self.device)

        print(f"Predictor initialized with model: {self.model_path} on device: {self.device}")


    def _load_model(self) -> torch.nn.Module:
        """
        Loads and prepares the model for prediction task.

        Returns
        -------
        torch.nn.Module
            Torch module in the form of a model architecture. The model weights are loaded on self.device
        """

        return self.loader_factory.load_model(self.model_loader_config.model_name)


    @torch.no_grad()
    def _predict_raw(self, X:torch.Tensor) -> torch.Tensor:
        """Calculates raw output from model

        Args:
            X (torch.Tensor): input tesor for prediction

        Returns:
            torch.Tensor: 1d tensor with raw outputs (not decoded) from model
        """
        
        # set to eval mode
        if self._model.training:
            self._model.eval()

        # check if devices match
        if self.device != X.device:
            X = X.to(self.device)
        
        return self._model(X).detach().cpu().flatten()


    @torch.no_grad()
    def _predict(self, X: torch.Tensor) -> torch.Tensor:
        """Calculates predictions for given tensor X and applies softmax to them

        Args:
            X (torch.Tensor): image tensor to predict from

        Returns:
            torch.Tensor: 1d tensor representing probabilities of given classes
        """

        return torch.softmax(self._predict_raw(X))

        
    def predict_from_dir(self, dir_path:str):
        #TODO: Create script that would call _predict for photos in given directory and them organize data in somekind of csv or plot 
        pass


class ModelLoaderConfig:
    """Holds all the necessary information for model retrieval like model paths, the target device, etc."""

    def __init__(self, device: torch.device, models_path: str, model_name: str) -> None:
        """
        Parameters
        ----------
        device : torch.device
            Device on to which load the model
        models_path : str
            Root path for all saved models
        model_name : str
            The name of the model to load as a file like resnet34.ptr for TorchModelLoader
            or resnet34 for TimmModelLoader

        Raises
        ------
        FileNotFoundError
            If the model location is not a file or not exists at all.
        """

        self.model_device: torch.device = device
        self.model_root_path: Path = Path(models_path)
        self.model_name = model_name
        self.model_path: Path = self.model_root_path / self.model_name

        if not self.model_path.exists():
            raise FileNotFoundError(f'provided path ({self.model_root_path=}) does not exist')

        if not self.model_path.is_file():
            raise FileNotFoundError(f'provided path ({self.model_root_path=}) is not a file')


class ModelLoader(ABC):
    """The Abstract Factory interface for model loading given the model_name and loader_config"""

    def __init__(self, loader_config: ModelLoaderConfig) -> None:
        """Saves loader_config for all model loaders to use"""
        self.loader_config: ModelLoaderConfig = loader_config


    @abstractmethod
    def load_model(self, model_name: str) -> torch.nn.Module:
        """Loads a given model"""


@final
class TimmModelLoader(ModelLoader):
    """Loader that uses timm in order to read and retrieve model weights"""

    def __init__(self, loader_config: ModelLoaderConfig) -> None:
        super().__init__(loader_config)

        self.model_cache_path: Path = Path(self.loader_config.model_root_path) / 'timm'


    @override
    def load_model(self, model_name: str) -> torch.nn.Module:
        """
        Retries and loads the pytorch model using timm library.

        Parameters
        ----------
        model_name : str
            The file name of the model to load must map to some model class.

        Returns
        -------
        model
            Pytorch Module contains the weights mapped to the appropriate device.
        """

        pretrained_config = PretrainedCfg(file=self.loader_config.model_path.absolute().as_posix(),
                                          source='file',
                                          custom_load=True)
        return timm.create_model(model_name,
                                 pretrained=True,
                                 pretrained_cfg=pretrained_config,
                                 cache_dir=self.model_cache_path).to(self.loader_config.model_device)


@final
class TorchModelLoader(ModelLoader):
    """Dummy interface for loading pytorch models intended for future use"""

    def __init__(self, loader_config: ModelLoaderConfig) -> None:
        super().__init__(loader_config)


    @override
    def load_model(self, model_name: str) -> torch.nn.Module:
        """
        Retries and loads the pytorch model by reading the weights from the appropriate file.

        Parameters
        ----------
        model_name : str
            The file name of the model to load must map to some model class.

        Returns
        -------
        model
            Pytorch Module contains the weights mapped to the appropriate device.

        Raises
        ------
        ValueError
            If the model_name is unrecognised.
        """

        match model_name:
            case 'cnn-1':
                model = ... # CNN1()

            case 'cnn-2':
                model = ... # CNN2()

            case _:
                raise ValueError(f'Unknown {model_name=} received')

        pretrained_weights = torch.load(self.loader_config.model_path,
                                        map_location=self.loader_config.model_device,
                                        weights_only=True)
        model.load_state_dict(pretrained_weights)

        return model


class ModelLoaderType(IntEnum):
    """Simple Enum holding all possible supported leader types"""

    TIMM_MODEL_FACTORY = 0
    TORCH_MODEL_FACTORY = auto()


def get_loader_factory(loader_type: ModelLoaderType, loader_config: ModelLoaderConfig) -> ModelLoader:
    """
    Function that instantiates the correct loader logic for retrival of machine learning model architectures.

    Parameters
    ----------
    loader_type : ModelLoaderType
        The information on to which loader to instantiate
    loader_config : ModelLoaderConfig
        The config for the model retrival

    Raises
    ------
    ValueError
        If the loader_type is unrecognised.
    """

    match loader_type:
        case ModelLoaderType.TIMM_MODEL_FACTORY:
            return TimmModelLoader(loader_config)

        case ModelLoaderType.TORCH_MODEL_FACTORY:
            return TorchModelLoader(loader_config)

    raise ValueError('Received unknown loader type')