import numpy as np
import matplotlib.pyplot as plt

# states = {'State 1': 0.22, 'State 2': 0.68, 'State 3': 0.1, 'State 4': 1}

def plot_states(counts):
    plt.bar(list(counts.keys()), counts.values(), color='#445db7', width=0.5)
    plt.ylabel('Probabilities')
    plt.show()

def normalise(counts): #Normalises the probabilities, assuming an input like the 'states' sample variable above
    total = sum(counts.values())
    return {k: v/total for k, v in counts.items()}

# plot_states(normalise(states))



# def simulate(state_vector, samples): #takes an input of a state vector and a no. of samples and simulates
#     plot_states()

# simulate([0,1,2,5], 1000)

# def measure_qbits():
    