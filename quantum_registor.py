import numpy as np
import mat_math_util

class quantum_registor ():
    
    def __init__ (self, size):
        self.size = size
        self.step = 0
        self.gates = []
        self.qbits = []
        for i in range(size):
            self.qbits.append([1,0])
    
    def set_initial_state (self, n, state):
        self.qbits[n] = state
    
    def state (self):
        '''Returns state vector of entangled qbits.'''
        return mat_math_util.vec_tensor_all(self.qbits)
    
    def normalized_probability (self):
        pass