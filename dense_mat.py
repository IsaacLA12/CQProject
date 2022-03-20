import numpy as np
from base_mat import base_mat, mat_type

class dense_mat (base_mat):
    '''
    ## Dense Matrix
    Simply a wrap of numpy 2d array.
    Implement base_mat methods for matrix calculation.
    '''
    
    def __init__ (self, mat:np.ndarray):
        '''Initialze dense matrix with a np 2d array.'''
        # validate mat input
        try:
            self.m, self.n = mat.shape
        except Exception:
            raise ValueError("Invalid matrix format.")
        # base mat implementation
        self.type = mat_type.dense
        self.mat = mat
    
    def get (self, i, j):
        return self.mat[i,j]
    
    def transpose(self):
        return self.mat.T
    
    def nparray (self):
        return self.mat