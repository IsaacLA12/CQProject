import numpy as np
from quantum_registor import quantum_registor
import matplotlib.pyplot as plt

def plot (x, p, title='', show=True):
    # plt.clf()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, p)
    ax.set_title(title)
    ax.set_xlabel('State')
    ax.set_ylabel('Probability')
    if show:
        fig.show()
    return fig

if __name__ == '__main__':
    qr = quantum_registor(5)
    x_lbl = [f'|{i}⟩' for i in range(qr.size_)]
    s = np.zeros(qr.size_)
    s[2] = 1
    plot(x_lbl, qr.measure(), title='init')
    qr.H_all()
    fig = plot(x_lbl, qr.measure(), title=f'Iteration 0')
    fig.savefig(f'fig/grover/iter_0.png')
    for i in range (1, 10):
        qr.grover()
        qr.oracle(s)
        fig = plot(x_lbl, qr.measure(), title=f'Iteration {i}', show=False)
        fig.savefig(f'fig/grover/iter_{i}.png')
    # plt.show()
    input()