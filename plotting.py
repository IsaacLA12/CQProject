import numpy as np
import matplotlib.pyplot as plt

# states = {'State 1': 0.22, 'State 2': 0.68, 'State 3': 0.1}

def plot_states(counts):
    plt.bar(list(counts.keys()), counts.values(), color='#445db7', width=0.5)
    plt.ylabel('Probabilities')
    plt.show()
    
# plot_states(states)


