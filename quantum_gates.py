import numpy as np

def oracle (n_bits, s):
    """Creates the oracle matrix"""
    size = int (2 ** n_bits)
    if len(s) != size:
        raise ValueError
    s_ket = np.reshape(np.array(s), (size, 1))
    return np.eye(size) - 2*s_ket@s_ket.T

def grover (n_bits):
    """Creates the grover matrix"""
    size = int (2 ** n_bits)
    psi_ket = np.reshape(np.full(size, 1/np.sqrt(size)), (size, 1))
    return 2*psi_ket@psi_ket.T - np.eye(size)

def qft(n_bits):
    """Creates the QFT matrix (nxn numpy array)"""
    n = int(2**n_bits)
    j = complex(0,1)
    Q = np.zeros((n,n),dtype=complex)
    W = np.exp(2*np.pi*j/n)
    for i in range(n):
        for k in range(n):
            Q[i][k] = W**(i*k)     
    return Q