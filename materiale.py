#Materiale

import numpy as np
import scipy as sp
import scipy.optimize as opt
import numpy.random as rnd

class isotope:
    def __init__(self, atomic_number=int, mass_number=int, density=float, energy=np.array, total=np.array, scattering=np.array, fission=None, chi = None):
        isotope.energy = energy
        isotope.zeta = atomic_number
        isotope.alpha = mass_number
        isotope.rho = density #g/cm3
        isotope.micro_total = total/(1E24)
        isotope.micro_scattering = scattering/(1E24)
        isotope.micro_absorption = isotope.micro_total-isotope.micro_scattering
        if fission is not None:
            isotope.micro_fission = fission/(1E24)
            isotope.fs_emission = chi
        else:
            isotope.micro_fission = np.zeros(len(energy))
            isotope.fs_emission = np.zeros(len(energy))
        isotope.micro_capture = isotope.micro_absorption-isotope.micro_fission
        isotope.atomic_density = 6E23*isotope.rho/isotope.alpha


class material:
    def __init__(self, parts=list[isotope]):
        nn = len(parts)
        material.energy = parts[0].energy
        material.macro_total = np.zeros(len(parts[0].micro_total))
        material.macro_scattering = np.zeros(len(parts[0].micro_total))
        material.macro_absorption = np.zeros(len(parts[0].micro_total))
        material.macro_fission = np.zeros(len(parts[0].micro_total))
        material.macro_capture = np.zeros(len(parts[0].micro_total))
        for ii in range(nn):
            material.macro_total += parts[ii].micro_total*parts[ii].atomic_density
            material.macro_scattering += parts[ii].micro_scattering*parts[ii].atomic_density
            material.macro_absorption += parts[ii].micro_absorption*parts[ii].atomic_density
            material.macro_fission += parts[ii].micro_fission*parts[ii].atomic_density
            material.macro_capture += parts[ii].micro_capture*parts[ii].atomic_density
    
    def xs_total(self, en=float):
        if en >= self.energy[-1]:
            out = self.macro_total[-1]
        elif en <= self.energy[0]:
            out = self.macro_total[0]
        else:
            index = np.where(self.energy>en)[0][0]
            out = np.interp(en, [self.energy[index-1],self.energy[index]], [self.macro_total[index-1],self.macro_total[index]])
        return out
    def xs_scattering(self, en=float):
        if en >= self.energy[-1]:
            out = self.macro_scattering[-1]
        elif en <= self.energy[0]:
            out = self.macro_scattering[0]
        else:
            index = np.where(self.energy>en)[0][0]
            out = np.interp(en, [self.energy[index-1],self.energy[index]], [self.macro_scattering[index-1],self.macro_scattering[index]])
        return out
    def xs_absorption(self, en=float):
        if en >= self.energy[-1]:
            out = self.macro_absorption[-1]
        elif en <= self.energy[0]:
            out = self.macro_absorption[0]
        else:
            index = np.where(self.energy>en)[0][0]
            out = np.interp(en, [self.energy[index-1],self.energy[index]], [self.macro_absorption[index-1],self.macro_absorption[index]])
        return out
    def xs_fission(self, en=float):
        if en >= self.energy[-1]:
            out = self.macro_fission[-1]
        elif en <= self.energy[0]:
            out = self.macro_fission[0]
        else:
            index = np.where(self.energy>en)[0][0]
            out = np.interp(en, [self.energy[index-1],self.energy[index]], [self.macro_fission[index-1],self.macro_fission[index]])
        return out
    def xs_capture(self, en=float):
        if en >= self.energy[-1]:
            out = self.macro_capture[-1]
        elif en <= self.energy[0]:
            out = self.macro_capture[0]
        else:
            index = np.where(self.energy>en)[0][0]
            out = np.interp(en, [self.energy[index-1],self.energy[index]], [self.macro_capture[index-1],self.macro_capture[index]])
        return out
    def xs_general(self, en=float):
        return [self.xs_scattering(en),self.xs_fission(en),self.xs_capture(en)]