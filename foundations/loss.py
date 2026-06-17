import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_pred+=1e-7 #smoothing
        summation = np.sum((y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred)))
        return np.round(-1/len(y_true)*summation,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_pred+=1e-7 #smoothing
        summation_c = 0
        for i in range(len(y_true)):
            y_true_i, y_pred_i = y_true[i], y_pred[i]
            summation_c+=y_true_i*np.log(y_pred_i)
        return np.round(-1/len(y_true)*np.sum(summation_c),4)
        
    

