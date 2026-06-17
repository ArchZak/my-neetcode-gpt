import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        
        #forward
        z = np.dot(x,w)+b #z
        z=1/(1+np.e**-z) #y hat
        #loss
        error = z-y_true
        sigma_z = z*(1-z) #sigma'(z)
        entropy = error*sigma_z

        return (np.round(x*entropy,5),np.round(entropy,5))

        


