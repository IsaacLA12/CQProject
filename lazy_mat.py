import numpy as np
from base_mat import base_mat, mat_type
from dense_mat import dense_mat

class lazy_mat (base_mat):
    def __init__ (self, left, right, op):
        '''
        
        '''
        self.type = mat_type.lazy
        self.left = left
        self.right = right
        self.op = op
        self.result = None
    
    def eval (self):
        '''
        Evaluate the lazy matrix, store and return result.
        '''
        if self.result == None:
            self.result = self.op(self.left, self.right, False)
        return self.result
    
    def get(self, i, j):
        self.eval()
        return self.result.get(i, j)
    
    def transpose(self):
        self.eval()
        return self.result.transpose()
    
    def nparray (self):
        self.eval()
        return self.result.nparray()
    
    def __str__(self) -> str:
        return f'<lazy matrix> \nleft={str(self.left)}, \nright={str(self.right)}, \nop={str(self.op)}'
        