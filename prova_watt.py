import numpy as np
import scipy as sp
import scipy.optimize as opt
import numpy.random as rnd
import matplotlib.pyplot as plt
import materiale as mt
import trasporto as tr

def watt(E,T,a):
    return np.exp(-E / T) * np.sinh(np.sqrt(a * E))
out = np.zeros(1000000)
for ii in range(len(out)):
    out[ii] += tr.sample_watt(0.988,2.249)
xx = np.linspace(0,20, 100)
yy = watt(xx, 1,2)

def placzek(x):
    return np.sin(x)/x
ics = np.logspace(0,9,100)
plt.plot(ics, placzek(ics))
plt.xscale('log')
plt.show()
