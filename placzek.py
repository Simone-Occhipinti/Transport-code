import numpy as np
import scipy.integrate as spint
import scipy.optimize as opt
import numpy.random as rnd
import matplotlib.pyplot as plt


def placzek1(aa):
    alfa = ((aa-1)/(aa+1))**2
    #xx = np.linspace(alfa*e0,e0,1000)
    #yy = ((e0/xx)**(alfa/(1-alfa)))/(xx*(1-alfa))
    xx = np.linspace(0,np.log(1/alfa),100)
    yy = (np.exp((alfa/(1-alfa))*xx))/(1-alfa)
    return [xx,yy]

def placzek2(aa):
    alfa = ((aa-1)/(aa+1))**2
    #xx = np.linspace(alfa*alfa*e0,e0*alfa,1000)
    #yy = ((e0/xx)**(alfa/(1-alfa)))/(xx*((1-alfa)**2))*(((1-alfa)*(1-alfa**(1/(1-alfa))))-((alfa**(alfa/(1-alfa))))*np.log(alfa*e0/xx))
    xx = np.linspace(np.log(1/alfa),np.log(1/alfa**2),100)
    yy = (1-alfa**(1/(1-alfa)))/(1-alfa)*np.exp(alfa/(1-alfa)*xx) - ((alfa**(alfa/(1-alfa)))/(1-alfa)**2)*(xx-np.log(1/alfa))*np.exp(alfa/(1-alfa)*xx)
    return [xx,yy]

def placzek3(aa):
    alfa = ((aa-1)/(aa+1))**2
    csi = 1+((alfa*np.log(alfa))/(1-alfa))
    #xx = np.linspace(1E-5,e0*alfa*alfa,1000)
    xx = np.linspace(np.log(1/alfa**2),6,100)
    yy = 1/csi*np.ones(len(xx))
    return [xx,yy]

def placzek(AA):
    part1 = placzek1(AA)
    part2 = placzek2(AA)
    part3 = placzek3(AA)

    xlab = np.hstack((part1[0],part2[0],part3[0]))
    #ylab = np.hstack((part3[1],solutions[0]*part2[1],solutions[0]*part1[1]))
    ylab = np.hstack((part1[1],part2[1],part3[1]))

    return [xlab,ylab]

#p1 = placzek1(16)
#p2 = placzek2(16,2E6)
#pp = placzek(12)
#S0 = spint.simpson(pp[1])

#print(spint.simpson(pp[1]))
#plt.plot(p1[0], p1[1])
#plt.plot(p2[0], p2[1])
#plt.plot(pp[0],pp[1])
#plt.xscale('log')
#plt.show()

def adj_placzek1(aa):
    alfa = ((aa-1)/(aa+1))**2
    #xx = np.linspace(alfa*e0,e0,1000)
    #yy = ((e0/xx)**(alfa/(1-alfa)))/(xx*(1-alfa))
    xx = np.linspace(2E6,2E6/alfa,100)
    yy = (1/(1-alfa))*(xx/2E6)**(1/(1-alfa)) 
    return [xx,yy]

def adj_placzek2(aa):
    alfa = ((aa-1)/(aa+1))**2
    #xx = np.linspace(alfa*alfa*e0,e0*alfa,1000)
    #yy = ((e0/xx)**(alfa/(1-alfa)))/(xx*((1-alfa)**2))*(((1-alfa)*(1-alfa**(1/(1-alfa))))-((alfa**(alfa/(1-alfa))))*np.log(alfa*e0/xx))
    xx = np.linspace(2E6/alfa,2E6/(alfa**2),100)
    yy = (1/(1-alfa)**2)*((xx/2E6)**(1/(1-alfa)))*(((1-alfa)*(1-alfa**(1/(1-alfa))))-((alfa**(1/(1-alfa)))*(np.log(alfa*xx/2E6))))
    return [xx,yy]

#def placzek3(aa):
#    alfa = ((aa-1)/(aa+1))**2
#    csi = 1+((alfa*np.log(alfa))/(1-alfa))
#    #xx = np.linspace(1E-5,e0*alfa*alfa,1000)
#    xx = np.linspace(np.log(1/alfa**2),6,100)
#    yy = 1/csi*np.ones(len(xx))
#    return [xx,yy]

def adj_placzek(AA):
    part1 = adj_placzek1(AA)
    part2 = adj_placzek2(AA)

    xlab = np.hstack((part1[0],part2[0]))
    #ylab = np.hstack((part3[1],solutions[0]*part2[1],solutions[0]*part1[1]))
    ylab = np.hstack((part1[1],part2[1]))

    return [xlab,ylab]

pp = adj_placzek(12)

plt.plot(pp[0],pp[1])
plt.xscale('log')
plt.show()