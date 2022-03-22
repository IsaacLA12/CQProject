import numpy as np
from quantum_registor import quantum_registor
import matplotlib.pyplot as plt
from math import gcd

def next_guess(g,p):
    """
    Produces the next best guess for prime factors in shor's algorithm given the previous guess g and period p
    """
    g1 = int(g**(p/2)+1)
    g2 = int(g**(p/2)-1)
    return g1, g2

def get_period(rems):
    """Given a vector with period elements, returns the period p"""
    n = len(rems) 
    k = rems[0] 
    for j in range(1,n):
        if rems[j]==k:
            p = j
            break
        else:
            continue
    return p

# simulation of N=221, g=12, p=16
if __name__ == '__main__':
    N = 221; g = 12
    qr = quantum_registor(8)
    qr.state = np.ones(qr.size_, dtype=complex)
    qr.shors_U(N, g)
    qr.measure()
    p = get_period(qr.state)
    print (f'{p=}')
    g1, g2 = next_guess(g, p)
    print (f'{g1=}\n{g2=}')
    a, b = gcd(N, g1), gcd(N, g2)
    print (f'gcd(N, g1)={a}\ngcd(N, g2)={b}')
    print (a * b)