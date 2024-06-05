#Particles

import numpy as np
import numpy.random as rnd
import scipy.integrate as spi
import materiale

class Cartesian:
    def __init__(self, ics=float, ips=float, zeta=float):
        self.x=ics
        self.y=ips 
        self.z=zeta
    @property    
    def distance(self):
        return np.sqrt(self.x**2+self.y**2+self.z**2)
    def cartesian2polar(self):
        if self.distance == 0:
            rr = 0
            teta = 0
            phi = 0
        else:
            rr = self.distance
            teta = np.arccos(self.z/rr)
            phi = np.arcsin(self.y/(rr*np.sin(teta)))
        return Polar(rr,teta,phi)
class Polar:
    def __init__(self, Rad=float,Teta=float, Phi=float):
        self.r = Rad
        self.teta = Teta
        self.phi = Phi
    def polar2cartesian(self):
        zz = self.r*np.cos(self.teta)
        xx = self.r*np.sin(self.teta)*np.cos(self.phi)
        yy = self.r*np.sin(self.teta)*np.sin(self.phi)
        return Cartesian(xx,yy,zz)

class Position:
    def __init__(self, ics=float, ips=float, zeta=float, type=str):
        if type == 'cartesian':
            self.cartesian = Cartesian(ics,ips,zeta)
            self.polar = self.cartesian.cartesian2polar()
        elif type == 'polar':
            self.polar = Polar(ics,ips,zeta)
            self.cartesian = self.polar.polar2cartesian()

class Direction:
    def __init__(self, Teta, Phi):
        self.teta = Teta
        self.phi = Phi
        self.mu = np.cos(self.teta)

class particle:
    def __init__(self, point=Position, angle=Direction, en=float, wgt=float, tp = str):
        self.position = point
        self.direction = angle
        self.energy = en
        self.weight = wgt
        self.type = tp

    def newposition(self, ll=float):
        self.position.cartesian.z += ll*np.cos(self.direction.teta)
        self.position.cartesian.x += ll*np.sin(self.direction.teta)*np.cos(self.direction.phi)
        self.position.cartesian.y += ll*np.sin(self.direction.teta)*np.sin(self.direction.phi)
        self.position.polar = self.position.cartesian.cartesian2polar()
    
    def sample_isotope(self, mat=materiale.material, parts=list[materiale.isotope], en=float):
        nn = len(parts)
        XS = np.zeros(nn)
        if en <= mat.energy[0]:
            index = 0
        elif en >= mat.energy[-1]:
            index = -1
        else:
            index = np.where(mat.energy>en)[0][0]
        for ii in range(nn):
            XS[ii] += parts[ii].micro_scattering[index]*parts[ii].atomic_density/mat.macro_scattering[index]
        rho = rnd.rand()
        cumulative = np.cumsum(XS)
        out = np.where(cumulative>rho)[0][0]
        return out
    
    def sample_freeflight(self, mat=materiale.material):
        rho = rnd.rand()
        ll = -1/mat.xs_total(self.energy)*np.log(rho)
        return ll
    
    def sample_interaction(self, mat=materiale.material,parts=list[materiale.material]):
        #implicit capture, no fission
        self.weight *= mat.xs_scattering(self.energy)/mat.xs_total(self.energy)

        if self.type == 'adjunction':
            ii = self.sample_isotope(mat,parts,self.energy)
            alfa = ((parts[ii].alpha-1)/(parts[ii].alpha+1))**2
            low_i = np.where(self.energy>=mat.energy)[0][-1]
            if self.energy/alfa <= mat.energy[-1]:
                up_i = np.where(self.energy/alfa<=mat.energy)[0][0]
            else:
                up_i = -1
            SS = np.trapz(mat.macro_scattering[low_i:up_i]/mat.energy[low_i:up_i], mat.energy[low_i:up_i])
            self.weight *= 1/(1-alfa)/mat.xs_scattering(self.energy)*SS

    def sample_direction(self):
        self.direction.mu = rnd.rand()*2-1
        self.direction.phi = 2*rnd.rand()-1
        self.direction.teta = np.arccos(self.direction.mu)

    def sample_energy(self, mat=materiale.material, parts=list[materiale.isotope]):
        # step function
        ii = self.sample_isotope(mat,parts,self.energy)
        alfa = ((parts[ii].alpha-1)/(parts[ii].alpha+1))**2

        if self.type == 'neutron':
            new_ee = alfa*self.energy+(1-alfa)*self.energy*rnd.rand()
            self.energy = new_ee

        elif self.type == 'adjunction':
            low_i = np.where(self.energy>=mat.energy)[0][-1]
            if self.energy/alfa <= mat.energy[-1]:
                up_i = np.where(self.energy/alfa<=mat.energy)[0][0]
            else:
                up_i = -1
            SS = np.trapz(mat.macro_scattering[low_i:up_i]/mat.energy[low_i:up_i], mat.energy[low_i:up_i])
            exponent = rnd.rand() * SS / mat.xs_scattering(self.energy)
            # Using logarithm to avoid overflow
            result_log = np.log(self.energy) + exponent
            if result_log>= np.log(mat.energy[-1]):
                self.energy = mat.energy[-1]+1
            else:
                self.energy = np.exp(result_log)

class estimator:
    def __init__(self, mm = materiale.material):
        self.count = []
        self.range = mm.energy
        self.ref = np.array([(self.range[ii]+self.range[ii+1])/2 for ii in range(len(self.range)-1)])   
        self.iter = 0
    
    def new_interaction(self, mm=materiale.material, nn = particle):
        index = np.where(nn.energy<=self.range)[0][0]
        self.count[self.iter-1][index-1] += nn.weight/mm.xs_scattering(self.ref[index-1])

    def new_iteration(self):
        self.count.append(np.zeros(len(self.ref)))
        self.iter += 1

    def sample_avg(self):
        avg = np.zeros(len(self.ref))
        for ii in range(self.iter):
            avg += self.count[ii]/self.iter
        
        mom2 = np.zeros(len(avg))
        for ii in range(self.iter):
            mom2 += (self.count[ii])**2/self.iter
        var = mom2 - avg**2
        return [avg, var]


def sample_watt(aa=float, bb=float):
    kk = 1 + bb/(8*aa)
    LL = (kk + np.sqrt(kk**2 - 1))/aa
    MM = aa*LL-1
    out = 0
    while out<=0:
        xx = -np.log(rnd.rand())
        yy = -np.log(rnd.rand())
        if ((yy-MM*(xx+1))**2)<=bb*LL*xx:
            out += LL*xx
    return out*1E6