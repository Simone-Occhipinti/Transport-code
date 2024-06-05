# transport code

import numpy as np
import scipy.integrate as spint
import numpy.random as rnd
import matplotlib.pyplot as plt
import materiale as mt
import trasporto as tr
import placzek

# tipo di particella: neutron, adjunction
simulation = 'neutron'
# tipo di sorgente: Fixed(2E6MeV), Watt (fission spectrum), given 
source = 'Fixed'
# numero di storie
NN = int(1E4)
# energia di riferimento
E0 = 2E6
# se si vuole includere la funzione di placzek
mode = 'Pl'

Cnat_total = np.loadtxt('cross_sections_Janis\Cnat_total.csv', delimiter=';',skiprows=3)
Cnat_scattering = np.loadtxt('cross_sections_Janis\Cnat_scattering.csv', delimiter=';', skiprows=3)

Onat_total = np.loadtxt('cross_sections_Janis\Onat_total.csv', delimiter=';',skiprows=3)
Onat_scattering = np.loadtxt('cross_sections_Janis\Onat_scattering.csv', delimiter=';', skiprows=3)

if mode == 'Pl':
    carbon = mt.isotope(6,12,2.267,Cnat_scattering[:,0],Cnat_scattering[:,1],Cnat_scattering[:,1])
    composition = [carbon]

else:
    oxigen = mt.isotope(8,16,4*1.4E-3,Onat_total[:,0],Onat_total[:,1],Onat_scattering[:,1])
    carbon = mt.isotope(6,12,2.267,Cnat_total[:,0],Cnat_total[:,1],Cnat_scattering[:,1])

    composition = [oxigen,carbon]

medium = mt.material(composition)

csi = tr.estimator(medium)
itermax = NN
wd = 5
wgt_min = 1/wd
wgt_max = wd

waiting_nn = []

jj = 0

#MC code
while csi.iter<itermax:
    # generazione nuova particella
    if len(waiting_nn) == 0:

        csi.new_iteration()

        if jj >= 100:
            print(csi.iter)
            jj-=100
        else:
            jj+=1
        
        eof = 1

        if source == 'Fixed':
            e0 = 2E6
        elif source == 'Watt':
            e0 = tr.sample_watt(0.988,2.249)
        else:
            e0 = source()
        phi0 = 2*np.pi*rnd.random()
        mu0 = 2*rnd.random()-1
        teta0 = np.arccos(mu0)
        p0 = tr.Position(0.,0.,0.,'cartesian')
        d0 = tr.Direction(teta0,phi0)

        nn = tr.particle(p0,d0,e0,1,simulation)
    # continuo della storia precedente
    else:
        nn = waiting_nn.pop(0)
        eof = 1

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
        
        # check for splitting
        if nn.weight>=wgt_max:
            N = nn.weight/wgt_max
            if N == int(N):
                for ii in range(N-1):
                    waiting_nn.append(tr.particle(nn.position, nn.direction, nn.energy, nn.weight/N, simulation))
                nn.weight *= 1/N
            else:
                D = N - int(N)
                if rnd.rand() <= 1-D:
                    for ii in range(int(N)-1):
                        waiting_nn.append(tr.particle(nn.position, nn.direction, nn.energy, nn.weight/int(N), simulation))
                    nn.weight *= 1/int(N)
                else:
                    for ii in range(int(N)):
                     waiting_nn.append(tr.particle(nn.position, nn.direction, nn.energy, nn.weight/(int(N)+1), simulation))
                     nn.weight *= 1/(int(N)+1)

# calvolo sample average
avg, var = csi.sample_avg()
sigma = np.sqrt(var/csi.iter)

# normalizzazione (DE/E0) + calcolo errrore
deltaE = np.diff(csi.range)
if nn.type == 'adjunction':
    deltaE *= 1/E0
    eflx = avg/(np.diff(csi.range)/E0)
else:
    eflx = avg/deltaE

if nn.type == 'neutron':
    plc = placzek.placzek(12)
elif nn.type == 'adjunction':
    plc = placzek.adj_placzek(12)
for ii in range(len(plc[1])):
    plc[1][ii] = plc[1][ii]/medium.xs_scattering(plc[0][ii])

eps1 = E0*((carbon.alpha-1)/(carbon.alpha+1))**2
eps2 = E0*((carbon.alpha-1)/(carbon.alpha+1))**4
eps4 = E0*((carbon.alpha-1)/(carbon.alpha+1))**16


plt.plot(csi.ref, eflx,  linewidth=4, label='MC results')
plt.plot(csi.ref,eflx-sigma,'o',markersize=2, c='black')
plt.plot(csi.ref,eflx+sigma,'o',markersize=2, c='black')

print(eflx-sigma)

plt.xlabel('energy [eV]')
if nn.type == 'neutron':
    plt.title('Direct transport problem')
    plt.ylabel(r'$\Phi$ [n/cm2/s]')
    if mode == 'Pl':
        plt.plot(plc[0],plc[1],linewidth=1.5,label='Placzek function')
        plt.axvline(x=eps1, color='r', linestyle='--', linewidth=0.5,label=r'$\alpha$ E0')
        plt.axvline(x=eps2, color='r', linestyle='--', linewidth=0.5,label=r'$\alpha^2$E0')
        plt.xlim(csi.range[1],2E6)
elif nn.type == 'adjunction':
    plt.title('Adjoint transport problem')
    plt.ylabel(r'$\Psi$ [-/cm2/s]')
    if mode == 'Pl':
        plt.plot(plc[0],plc[1],linewidth=1.5,label='Placzek function')
        plt.xlim(2E6,2E8)
else:
    plt.ylabel('errore')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()