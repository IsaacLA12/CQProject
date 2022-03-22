import numpy as np
import mat_math_util

from dense_mat import dense_mat
from sparse_mat import sparse_mat

class quantum_registor ():
    
    def __init__ (self, size):
        self.size = size; self.size_ = int(2**size)
        self.steps = []
        self.matrix = mat_math_util.identity(size)
        # initialize the state vector
        qbits = []
        for i in range(size):
            qbits.append([1,0])
        self.state = mat_math_util.vec_tensor_all(qbits)
    
    def H_all (self):
        '''H-gate to all qbits.'''
        h = dense_mat((1/np.sqrt(2)) * np.array([[1, 1], [1, -1]]))
        m = dense_mat(np.array([[1]]))
        for i in range (self.size):
            m = mat_math_util.mat_tensor(h, m)
        self.steps.append(m)
        # self.matrix = mat_math_util.mat_mul(self.matrix, m)
    
    def grover (self):
        '''The input s should be a numpy 1d array'''
        psi_ket = np.full((self.size_, 1), 1/np.sqrt(self.size_))
        m = 2*psi_ket@psi_ket.T - np.eye(int(2**self.size))
        self.steps.append(sparse_mat(m))
        # self.matrix = mat_math_util.mat_mul(self.matrix, sparse_mat(m))
    
    def oracle (self, s:np.ndarray):
        '''The input s should be a numpy 1d array'''
        s_ket = np.reshape(s, (len(s), 1))
        m = np.eye(int(2**self.size)) - 2*s_ket@s_ket.T
        self.steps.append(sparse_mat(m))
        # self.matrix = mat_math_util.mat_mul(self.matrix, sparse_mat(m))
        
    def shors_U (self, N, g):
        self.steps.append(('shors_U', N, g))
    
    def qft (self):
        """Creates the QFT matrix (nxn numpy array)"""
        n = self.size_
        j = complex(0,1)
        Q = np.zeros((n,n),dtype=complex)
        W = np.exp(2*np.pi*j/n)
        for i in range(n):
            for k in range(n):
                Q[i][k] = W**(i*k)     
        self.steps.append(dense_mat(Q))

    def apply_mat (self):
        self.state = mat_math_util.apply(self.matrix, self.state)
        self.matrix = mat_math_util.identity(self.size)
    
    def measure (self):
        """Measure the quantum states. Activate the matrix calculation, and update the state vector."""
        for step in self.steps:
            if type(step) == tuple:
                self.apply_mat()
                if step[0] == 'shors_U':
                    _, n, g = step
                    for i in range(self.size_):
                        el = (g**(i+1))%n
                        self.state[i] = el
                        # print (el)
            else:
                self.matrix = mat_math_util.mat_mul(self.matrix, step)
        self.steps = []
        self.apply_mat()
        
        norm_p = self.state**2
        norm_p *= 1 / np.sum(norm_p) # normalize
        return norm_p
    
    
    # qr = quantum_registor(3)
    # s = np.zeros(int(2**qr.size))
    # s[2] = 1
    # qr.H_all()
    # qr.G()
    # qr.O(s)
    # # S2
    # qr.G()
    # qr.O(s)
    # qr.measure()
    