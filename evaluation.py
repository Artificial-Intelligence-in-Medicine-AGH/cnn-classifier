from pydantic import BaseModel
from typing import List, Union
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np


class MetricsOutput(BaseModel):
    precision: float
    recall: float

class Evaluation():
    def precision_recall(self, y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> MetricsOutput:
        #walidacja ksztaltów
        if len(y_true) != len(y_pred):
            raise ValueError(f"Shape mismatch: y_true ma długość {len(y_true)}, a y_pred {len(y_pred)}")

        
        #klasyfikacja binarna czy multiclass
        unique_classes = set(y_true)
        
        if len(unique_classes) > 2:
            avg_param = 'weighted' 
        else:
            avg_param = 'binary'

        
        # zero_division=0 zeby nie dzielic przez zero
        precision = precision_score(y_true, y_pred, average=avg_param, zero_division=0)
        recall = recall_score(y_true, y_pred, average=avg_param, zero_division=0)

        
        return MetricsOutput(precision=precision, recall=recall)

    def accuracy(self, y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]) -> float:
        # input shape validation
        if len(y_true) != len(y_pred):
            raise ValueError(f"Shape mismatch: y_true has length {len(y_true)}, "f"but y_pred has length {len(y_pred)}")

        accuracy = accuracy_score(y_true, y_pred)
        return float(accuracy)