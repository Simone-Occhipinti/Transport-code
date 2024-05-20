# transport code

import numpy as np
import scipy as sp
import scipy.optimize as opt
import numpy.random as rnd
import matplotlib.pyplot as plt
import materiale as mt
import trasporto as tr

#input data
Cnat_total = np.loadtxt('cross_sections_Janis\Cnat_total.csv', delimiter=';',skiprows=3)
Cnat_scattering = np.loadtxt('cross_sections_Janis\Cnat_scattering.csv', delimiter=';', skiprows=3)

Onat_total = np.loadtxt('cross_sections_Janis\Onat_total.csv', delimiter=';',skiprows=3)
Onat_scattering = np.loadtxt('cross_sections_Janis\Onat_scattering.csv', delimiter=';', skiprows=3)

oxigen = mt.isotope(8,16,4*1.4E-3,50*Onat_total[:,0],Onat_total[:,1],Onat_scattering[:,1])
carbon = mt.isotope(6,12,2.267,50*Cnat_total[:,0],Cnat_total[:,1],Cnat_scattering[:,1])

#oxigen = mt.isotope(8,16,4*1.4E-3,Onat_scattering[:,0],Onat_scattering[:,1],Onat_scattering[:,1])
#carbon = mt.isotope(6,50,2.267,Cnat_scattering[:,0],Cnat_scattering[:,1],Cnat_scattering[:,1])

composition = [oxigen,carbon]
#composition = [carbon]
medium = mt.material(composition)

csi = tr.estimator(medium, 2E7, 1000)
itermax = 1000

wd = 5
wgt_min = 1/wd
wgt_max = wd

jj = 0
while csi.iter<itermax:
    # inizio una nuova storia
    csi.new_iteration()
    eof = 1

    if jj >= 100:
        print(csi.iter)
        jj-=100
    else:
        jj+=1

    # genero una nuova particella dallo spettro di fissione (Watt)
    e0 = tr.sample_watt(0.988,2.249)
    #e0 = 2E6
    phi0 = 2*np.pi*rnd.random()
    mu0 = 2*rnd.random()-1
    teta0 = np.arccos(mu0)
    p0 = tr.Position(0.,0.,0.,'cartesian')
    d0 = tr.Direction(teta0,phi0)

    nn = tr.particle(p0,d0,e0,1)
    tt = 0
    # simulo il trasporto
    while eof > 0:
        # distanza per l'interazione
        flight = nn.sample_freeflight(medium)
        # conto l'interazione
        csi.new_interaction(medium,nn)
        
        # modifico l'angolo e l'energia
        nn.sample_direction()
        nn.sample_energy(medium,composition)
        
        # aggiorno il peso dopo l'interazione
        nn.sample_interaction(medium)

        # check for russian roulette
        if nn.weight<=wgt_min:
            rho = rnd.rand()
            if rho<=1/wd:
                nn.weight *= wd
            else:
                eof = 0
        # energy check
        if nn.energy<=medium.energy[0] or nn.energy>=medium.energy[-1]:
            #rho = rnd.rand()
            #if rho<=1/wd:
            #    nn.energy *= wd
            #else:
            eof = 0
        #if eof == 0:
        #    plt.plot(csi.ref[0:maxenergy], csi.count[csi.iter-1][0:maxenergy])
        #    plt.show()

avg = csi.sample_avg()

#fit_solution = np.poly1d(np.polyfit(csi.ref, avg, 8))
#lethargy = -np.log(np.array(csi.ref)/2E6)
#plt.plot(lethargy, avg)
plt.plot(csi.ref, avg)
#plt.xscale('log')
#plt.yscale('log')
plt.show()


