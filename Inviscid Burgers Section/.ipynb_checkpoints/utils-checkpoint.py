import numpy as np
import tensorflow
from numba import jit

@jit(nopython=True)
def mse(y,yy):return np.mean(np.power(y-yy,2))

def recursive_prediction(model,inp,K,grid_size):
    '''
    model : Deep learning model
    inp   : Input solution for the model
    K     : iterations for the deep learning model
    '''
    solutions =np.zeros((1,K,grid_size,grid_size,1))
    solutions[:,[0]] = model.predict(inp)
    for ki in range(1,K): solutions[:,[ki]] = model.predict(solutions[:,[ki-1]])
    return solutions 