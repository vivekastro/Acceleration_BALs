#!/usr/bin/python
#-*- coding: utf-8 -*-

import scipy
import numpy as np

class Spectrum(object):
    '''
    The Spectrum object
    '''
    def __init__(self):
        '''
        Initialize spectrum object
        '''
        self._wavelengths= None
        self._flux = None
        self._flux_error = None
        self._c = None
        self._units = None
        
    @property
    def wavelengths(self):
        return self._wavelengths
        
    @wavelengths.setter
    def wavelengths(self, new_wave):
        self._wavelengths = new_wave
        
    @property
    def flux(self):
        return self._flux
        
    @flux.setter
    def flux(self, new_flux):
        self._flux = new_flux

    @property
    def flux_error(self):
        return self._flux_error
        
    @flux_error.setter
    def flux_error(self, new_ferr):
        self._flux_error = new_ferr

    @property
    def c(self):
        return self._c
        
    @c.setter
    def c(self, new_c):
        self._c = new_c

    @property
    def units(self):
        return self._units
        
    @units.setter
    def units(self, new_units):
        self._units = new_units

    @property
    def redshift(self):
        return self._redshift
        
    @units.setter
    def redshift(self, redshift):
        self._redshift = new_redshift

    @property
    def objname(self):
        return self._objname
        
    @units.setter
    def objname(self, objname):
        self._objname = new_objname










