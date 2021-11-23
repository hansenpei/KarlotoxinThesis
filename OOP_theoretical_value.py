# -*- coding: utf-8 -*-
"""
OOP Module for calculating the theoretical speed

@author: Hansen Pei
"""
#% import everything and initialization
import numpy as np
#from scipy.interpolate import interp1d
#import scipy.integrate as integrate

#% Class Definition
class s_module:
    """
    The class for calculating speed s given any x
    """
    def __init__(self, gamma = 0.532154, beta = 0.2, kappa = 0.2, 
                 fourier_N = 60, a = 3, b = 17, domain_size = 1):
        self.gamma = gamma # need to change to 0.5 for finite diff methods
        self.beta = beta
        self.kappa = kappa
        self.fourier_N = fourier_N # number of fourier terms
        self.a = a
        self.b = b
        self.domain_size = domain_size
        
    def c(self, x):
        """
        calculate the unbounded toxin concentration
        """
        r1 = (-1 + np.sqrt(1+4*self.beta*self.kappa))/(2*self.kappa)
        r2 = (-1 - np.sqrt(1+4*self.beta*self.kappa))/(2*self.kappa)
        unbounded_c = (np.exp(r1*(x-self.gamma))*np.heaviside(-x+self.gamma, 1/2) +\
                      np.exp(r2*(x-self.gamma))*np.heaviside(x-self.gamma, 1/2)) /\
                        np.sqrt(1+4*self.beta*self.kappa)
        return unbounded_c
    
    def c_periodic(self, x):
        """
        calculate the periodic toxin concentration, 
        as a sum of unbounded concentration
        """
        periodic_c = 0
        for n in range(-self.fourier_N, self.fourier_N+1):
            periodic_c += self.c(x + n*self.domain_size)
        return periodic_c
    
    def speed(self, x):
        """
        Calculate v according to the toxin concentration
        Linear reverse: -a*c(x) + b
        """
        spot_speed = -self.a*self.c_periodic(x) + self.b
        return spot_speed
