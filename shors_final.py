import numpy as np
import random
import matplotlib.pyplot as plt
import math as m

def qft(n):
    """
    Creates the QFT matrix (nxn numpy array)
    """
    
    j = complex(0,1)
    
    Q = np.zeros((n,n),dtype=complex)

    W = np.exp(2*np.pi*j/n)
    
    for i in range(n):
        for k in range(n):
            Q[i][k] = W**(i*k)
            
    return Q

def init(n):
    """
    To be replaced by np.sqrt(2**n)*Hadamard initialised state
    """
    
    v = []
    for i in range(n):
        v.append(1)
        
    return np.array(v,dtype=complex)

def U(v,g):
    """
    Creates adn applies the modular exponentiation matrix (nxn numpy array)
    to vector v (nx1 numpy array)
    """
    
    n = v.shape[0]
    
    for i in range(n):
        el = (g**(i+1))%n
        v[i] = el
    
    return v

def measure(v,s):
    """
    Given a specified value s, searches through vector v for each instance
    of the number s
    
    Returns a vector containing only the values s, other elements set to
    zero
    """
    
    n = v.shape[0]
    
    vect = v.tolist()
    
    for i in range(n):
        if vect[i]!=s:
            vect[i] = 0
        else:
            continue
    
    return np.array(vect)

def plot_qft(n,g):
    """
    From a given dimension n and a guess g, computes and applies the
    quantum fourier transform and then plots the probability of measuring
    each state
    """
    
    vect = init(n)                      # initialised state
    rems = U(vect,g)                    # finds remainders via modular exponentiation
    print (rems)
    exit()
    print(f"remainders = {make_list_real(rems)}")
    
    s = random.choice(rems.tolist())    # makes a random choice of remainder to measure
        
    print(f"remainder chosen = {int(np.real(s))}")
    
    spec_rems = measure(rems,s)         # measures s
    
    print(f"states that give remainder {int(np.real(s))} = {make_list_real(spec_rems)}")
    
    Q = qft(n)                          # creates qft matrix
    
    Q_app = np.matmul(Q,spec_rems)      # applies qft matrix to measured state
    
    xdata = []
    
    Q_list = Q_app.tolist()             # array -> list
    
    ydata = []
    
    for j in range(n):
        y = np.conjugate(Q_list[j])*Q_list[j]   # ydata = probabilities
        ydata.append(y)
    
    ydata = normalise(ydata)                    # normalise probabilities to sum to unity
    
    for i in range(n):
        xdata.append(i)                         # xdata = states
    
    fig, ax = plt.subplots()             # plot probabilities against states
    ax.plot(xdata, ydata)
    plt.title("Probability of Measurement vs State")
    plt.xlabel("State")
    plt.ylabel("Probability")
    plt.show()

    
    print(f"p from graph = {p_from_graph(xdata, ydata)[0]}")
    
    p = get_period(rems)                        # manually finds the period from numbers

    print(f"p = {p}")

def make_list_real(ls):
    """
    Extracts the real part of every element of a list
    """
    
    new_ls = []
    
    for l in ls:
        l_new = int(np.real(l))
        new_ls.append(l_new)
    
    return new_ls

def get_period(rems):
    """
    Given a vector with period elements, returns the period p
    """
    
    n = len(rems)
        
    k = rems[0]
    
    for j in range(1,n):
        
        if rems[j]==k:
            p = j
            break
        else:
            continue
    
    return p

def p_from_graph(xdata,ydata):
    """
    Finds the period of a sharply-peaked periodic function from a graph
    """
    n = len(ydata)
    
    peaks_y = []
    peaks_x = []
    
    for i in range(n):                     # extracts the (x,y) values for each point that has a y-value above a specified threshold, here 0.1
        if ydata[i]>0.1:
            peaks_y.append(ydata[i])
            peaks_x.append(xdata[i])
        else:
            continue
        
    m = len(peaks_x)
        
    diffs = []                             
    
    for j in range(1,m):                   # calculates the difference between the x-values of adjacent peaks, ignoring peaks that are two close together (e.g. multiple data points on one peak)
        x_n1 = peaks_x[j-1]
        x_n = peaks_x[j]
        if np.abs(x_n-x_n1)<3:
            continue
        else:
            diff = np.abs(x_n-x_n1)
            diffs.append(diff)
        
    l = len(diffs)
    
    ave_diffs = 0
    
    for d in diffs:                        # averages over the differences to get the average period
        ave_diffs += d
        
    ave_diffs = ave_diffs/l
    
    return ave_diffs, peaks_x, peaks_y

def normalise(vect):
    """
    Normalises a vector (list)
    """
    
    mag = 0
    
    n = len(vect)
    
    for i in range(n):
        v = vect[i]
        mag += np.conjugate(v)*v
    
    norm = 1/(mag**(1/2))
    
    for j in range(n):
        vect[j] = vect[j]*norm
    
    return vect

def next_guess(g,p):
    """
    Produces the next best guess for prime factors in shor's algorithm given the previous guess g and period p
    """
    g1 = int(g**(p/2)+1)
    g2 = int(g**(p/2)-1)
    return g1, g2
        
# simulation of N=221, g=12, p=16
if __name__ == '__main__':
    print("N=221, g=12")

    print(plot_qft(221,12))

    g1, g2 = next_guess(12,16)

    print("next guess:")
    print(f"g^(p/2)+1 = {g1}")
    print(f"g^(p/2)-1 = {g2}")

    gcd_1 = m.gcd(221,g1)
    gcd_2 = m.gcd(221,g2)

    print(f"gcd(221,{g1}) = {gcd_1}")
    print(f"gcd(221,{g2}) = {gcd_2}")

    print(f"gcd(221,{g1})*gcd(221,{g2}) = {gcd_1}*{gcd_2} = {gcd_1*gcd_2} = N")

# print (qft(16))

