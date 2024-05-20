#Particles

import numpy as np
import numpy.random as rnd
import scipy as sp
import scipy.optimize as opt
import materiale

class Cartesian:
    def __init__(self, ics=float, ips=float, zeta=float):
        self.x=ics
        self.y=ips 
        self.z=zeta
    def distance(self):
        return np.sqrt(self.x**2+self.y**2+self.z**2)
    def cartesian2polar(self):
        if self.distance() == 0:
            rr = 0
            teta = 0
            phi = 0
        else:
            rr = self.distance()
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
    def __init__(self, point=Position, angle=Direction, en=float, wgt=float):
        self.position = point
        self.direction = angle
        self.energy = en
        self.weight = wgt
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
    
    def sample_interaction(self, mat=materiale.material):
        #implicit capture, no fission
        self.weight *= mat.xs_scattering(self.energy)/mat.xs_total(self.energy)

    def sample_direction(self):
        self.direction.mu = rnd.rand()*2-1
        self.direction.phi = 2*rnd.rand()-1
        self.direction.teta = np.arccos(self.direction.mu)

    def sample_energy(self, mat=materiale.material, parts=list[materiale.isotope]):
        # step function
        ii = self.sample_isotope(mat,parts,self.energy)
        alfa = ((parts[ii].alpha-1)/(parts[ii].alpha+1))**2
        new_ee = alfa*self.energy+(1-alfa)*self.energy*rnd.rand()
        self.energy = new_ee

class estimator:
    def __init__(self, mm = materiale.material, max_en=float,nn=int):
        self.count = []
        #self.range = mm.energy
        self.range = np.linspace(mm.energy[0], max_en, nn)        
        self.ref = np.array([(self.range[ii]+self.range[ii+1])/2 for ii in range(len(self.range)-1)])
        self.iter = 0
    
    def new_interaction(self, mm=materiale.material, nn = particle):
        index = np.where(nn.energy<=self.range)[0][0]
        self.count[self.iter-1][index-1] += nn.weight/mm.xs_total(self.ref[index-1])
        #self.count[self.iter-1][index-1] += nn.weight

    def new_iteration(self):
        self.count.append(np.zeros(len(self.ref)))
        self.iter += 1

    def sample_avg(self):
        total = np.zeros(len(self.ref))
        avg = np.zeros(len(self.ref))
        for ii in range(self.iter):
            total += self.count[ii]
        for jj in range(len(self.ref)):
            avg[jj] += total[jj]/self.iter/(self.range[jj+1]-self.range[jj])
        return avg

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