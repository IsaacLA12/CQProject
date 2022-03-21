import numpy as np
import mat_math_util
import matplotlib.pyplot as plt

from dense_mat import dense_mat
from sparse_mat import sparse_mat
from lazy_mat import lazy_mat

class quantum_registor ():
    
    def __init__ (self, size):
        self.size = size
        self.size_sq = int(2**size)
        self.matrix = mat_math_util.identity(size)
        qbits = []
        for i in range(size):
            qbits.append([1,0])
        self.state = mat_math_util.vec_tensor_all(qbits)
        
    
    def set_initial_state (self, n, state):
        self.qbits[n] = state
    
    def H_all (self):
        '''H gate to all qbits. Initialization only.'''
        h = dense_mat((1/np.sqrt(2)) * np.array([[1, 1], [1, -1]]))
        m = dense_mat(np.array([[1]]))
        for i in range (self.size):
            m = mat_math_util.mat_tensor(h, m)
        self.matrix = mat_math_util.mat_mul(self.matrix, m)
    
    def G (self):
        '''The input s should be a numpy 1d array'''
        psi_ket = np.reshape(np.full(self.size_sq, 1/np.sqrt(self.size_sq)), (len(s), 1))
        m = 2*psi_ket@psi_ket.T - np.eye(int(2**self.size))
        self.matrix = mat_math_util.mat_mul(self.matrix, sparse_mat(m))
    
    def O (self, s:np.ndarray):
        '''The input s should be a numpy 1d array'''
        s_ket = np.reshape(s, (len(s), 1))
        m = np.eye(int(2**self.size)) - 2*s_ket@s_ket.T
        self.matrix = mat_math_util.mat_mul(self.matrix, sparse_mat(m))
    
    def measure (self):
        self.state = mat_math_util.apply(self.matrix, self.state)
        self.matrix = mat_math_util.identity(self.size)
        norm_p = self.state**2
        norm_p *= 1 / np.sum(norm_p) # normalize
        # plotting
        s_ = [f'|{i}⟩' for i in range(int(2**self.size))]
        plt.bar(s, norm_p)
        plt.xlabel('State')
        plt.ylabel('P')
        plt.show()

def test_g(psi, n=3):
    psi_ket = np.reshape(np.array(psi), (len(psi), 1))
    return 2*psi_ket@psi_ket.T - np.eye(int(2**n))
    
def test_o(s, n=3):
    s_ket = np.reshape(np.array(s), (len(s), 1))
    return np.eye(int(2**n)) - 2*s_ket@s_ket.T



if __name__ == '__main__':
    n = 5; sz = int(2**n)
    s = np.zeros(sz)
    s[2] = 1
    states = [np.full(sz, 1/np.sqrt(sz))]
    # print (test_o(s))
    # print (test_g(states[0]))
    s_ = [f'|{i}⟩' for i in range(sz)]
    for i in range(10):
        p = states[-1]**2
        p /= np.sum(p)
        plt.figure(1, (12,5))
        plt.title(f'Iteration {i}')
        plt.bar(s_, p)
        plt.savefig(f'{n}bit_iter_{i}')
        plt.clf()

        state = states[-1]
        state_ket = np.reshape(np.array(state), (len(state), 1))
        result = test_g(states[0], n=n)@test_o(s, n=n)@state_ket
        states.append(result.flatten())
    
    
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
    