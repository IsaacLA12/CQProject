import numpy as np
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
from dense_mat import dense_mat
=======
>>>>>>> parent of 0acbc55 (update)
=======
>>>>>>> parent of 0acbc55 (update)
=======
>>>>>>> parent of 0acbc55 (update)
=======
>>>>>>> parent of f206243 (1)
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
    
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    def h_all (self):
        h = dense_mat((1 / np.sqrt(2)) * np.array([[1, 1,], [1, -1]]))
        m = h
        for i in range(self.size - 1):
            m = mat_math_util.mat_tensor(h, m)
        print (mat_math_util.apply(m, self.state()))

    def h (self, n):
        '''Apply hadamard to nth qbit.'''
        pass

=======
>>>>>>> parent of 0acbc55 (update)
=======
>>>>>>> parent of 0acbc55 (update)
=======
>>>>>>> parent of 0acbc55 (update)
=======
>>>>>>> parent of f206243 (1)
    def state (self):
        '''Returns state vector of entangled qbits.'''
        return mat_math_util.vec_tensor_all(self.qbits)
    
    def normalized_probability (self):
        pass