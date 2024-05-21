# transport code

import numpy as np
import scipy.integrate as spint
import numpy.random as rnd
import matplotlib.pyplot as plt
import materiale as mt
import trasporto as tr
import placzek

#input data
Cnat_total = np.loadtxt('cross_sections_Janis\Cnat_total.csv', delimiter=';',skiprows=3)
Cnat_scattering = np.loadtxt('cross_sections_Janis\Cnat_scattering.csv', delimiter=';', skiprows=3)

Onat_total = np.loadtxt('cross_sections_Janis\Onat_total.csv', delimiter=';',skiprows=3)
Onat_scattering = np.loadtxt('cross_sections_Janis\Onat_scattering.csv', delimiter=';', skiprows=3)

#oxigen = mt.isotope(8,16,4*1.4E-3,Onat_total[:,0],Onat_total[:,1],Onat_scattering[:,1])
#carbon = mt.isotope(6,12,2.267,Cnat_total[:,0],Cnat_total[:,1],Cnat_scattering[:,1])

#oxigen = mt.isotope(8,16,4*1.4E-3,Onat_scattering[:,0],Onat_scattering[:,1],Onat_scattering[:,1])
carbon = mt.isotope(6,12,2.267,Cnat_scattering[:,0],Cnat_scattering[:,1],Cnat_scattering[:,1])

#composition = [oxigen,carbon]
composition = [carbon]
medium = mt.material(composition)

csi = tr.estimator(medium)
itermax = 20000
#E0 = 20E6
E0 = 2E6
wd = 5
wgt_min = 1/wd
wgt_max = wd

jj = 0

#MC code
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
    #e0 = tr.sample_watt(0.988,2.249)
    e0 = 2E6
    phi0 = 2*np.pi*rnd.random()
    mu0 = 2*rnd.random()-1
    teta0 = np.arccos(mu0)
    p0 = tr.Position(0.,0.,0.,'cartesian')
    d0 = tr.Direction(teta0,phi0)

    nn = tr.particle(p0,d0,e0,1,'adjunction')
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
        nn.sample_interaction(medium, composition)

        # check for russian roulette
        if nn.weight<=wgt_min:
            rho = rnd.rand()
            if rho<=1/wd:
                nn.weight *= wd
            else:
                eof = 0
        # energy check
        if nn.energy<=medium.energy[0] or nn.energy>=medium.energy[-1]:
            eof = 0

avg = csi.sample_avg()

lethargy = np.log(E0/csi.range)
plotLet = np.array([(lethargy[ii]+lethargy[ii+1])/2 for ii in range(len(lethargy)-1)])
deltaLet = np.diff(lethargy)
eflx = avg/(-deltaLet)

var = np.zeros(len(eflx))
for ii in range(len(var)):
    for jj in range(csi.iter):
        var[ii] += ((csi.count[jj][ii]/deltaLet[ii] - eflx[ii])**2)/(csi.iter-1)
sigma = (np.sqrt(var/csi.iter))

#plc = placzek.placzek(12)
plc = placzek.adj_placzek(12)
#for ii in range(len(plc[1])):
    #plc[1][ii] = plc[1][ii]/medium.xs_scattering(E0*np.exp(-plc[0][ii]))
#    plc[1][ii] = plc[1][ii]/medium.xs_scattering(plc[0][ii])

eps1 = E0*((carbon.alpha-1)/(carbon.alpha+1))**2
eps2 = E0*((carbon.alpha-1)/(carbon.alpha+1))**4
eps4 = E0*((carbon.alpha-1)/(carbon.alpha+1))**16

#plt.plot(plotLet, eflx,  linewidth=4, label='MC results')
#plt.plot(plotLet,eflx-sigma,'o',markersize=1, c='black')
#plt.plot(plotLet,eflx+sigma,'o',markersize=1, c='black')

plt.plot(csi.ref, eflx,  linewidth=4, label='MC results')
plt.plot(csi.ref,eflx-sigma,'o',markersize=1, c='black')
plt.plot(csi.ref,eflx+sigma,'o',markersize=1, c='black')

plt.plot(plc[0],plc[1],linewidth=1.5,label='Placzek function')

#plt.axvline(x=eps1, color='r', linestyle='--', linewidth=0.5,label=r'$\alpha$ E0')
#plt.axvline(x=eps2, color='r', linestyle='--', linewidth=0.5,label=r'$\alpha^2$E0')

#plt.axvline(x=np.log(1/((carbon.alpha-1)/(carbon.alpha+1))**2), color='r', linestyle='--', linewidth=0.5,label=r'$\alpha$ E0')
#plt.axvline(x=np.log(1/((carbon.alpha-1)/(carbon.alpha+1))**4), color='r', linestyle='--', linewidth=0.5,label=r'$\alpha^2$E0')

#plt.title('Direct transport problem')
plt.title('Adjoint transport problem')


#plt.xlabel('lethargy [-]')
plt.xlabel('energy [eV]')
#plt.ylabel(r'$\Phi$ [n/cm2/s/eV]')
plt.ylabel(r'$\Sigma_s\Psi$ [-/cm3/s]')
#plt.ylabel(r'$\Psi [-/cm2/s]$')
plt.xlim(2E6,20E6)
plt.xlim(E0)
plt.legend()
plt.xscale('log')
#plt.yscale('log')
plt.show()