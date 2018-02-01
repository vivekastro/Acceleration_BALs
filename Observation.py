#!/usr/bin/python
#-*- coding: utf-8 -*-

import scipy
import numpy as np

class Observation(object):
    '''
    The Observation object
    '''
    def __init__(self):
        '''
        Initialize Observation object
        '''
        self._pmf= None
        self._snr = None
        self._survey = None
        self._filename = None
       
    @property
    def pmf(self):
        return self._pmf
        
    @pmf.setter
    def pmf(self, new_pmf):
        self._pmf = new_pmf
        
    @property
    def snr(self):
        return self._snr
        
    @snr.setter
    def snr(self, new_snr):
        self._snr = new_snr

    @property
    def survey(self):
        return self._survey
        
    @survey.setter
    def survey(self, new_survey):
        self._survey = new_survey

    @property
    def filename(self):
        return self._filename
        
    @filename.setter
    def filename(self, new_filename):
        self._filename = new_filename
 
    @property
    def redshift(self):
        return self._redshift
        
    @redshift.setter
    def redshift(self, new_redshift):
        self._redshift = new_redshift

    @property
    def bi(self):
        return self._bi
        
    @bi.setter
    def bi(self, new_bi):
        self._bi = new_bi

    @property
    def bi_err(self):
        return self._bi_err
        
    @bi_err.setter
    def bi_err(self, new_bi_err):
        self._bi_err = new_bi_err

    @property
    def v_max(self):
        return self._v_max
        
    @v_max.setter
    def v_max(self, new_v_max):
        self._v_max = new_v_max

    @property
    def v_min(self):
        return self._v_min
        
    @v_min.setter
    def v_min(self, new_v_min):
        self._v_min = new_v_min

    @property
    def mjd(self):
        return self._mjd
        
    @mjd.setter
    def mjd(self, new_mjd):
        self._mjd = new_mjd

    @property
    def spec_name(self):
        return self._spec_name
        
    @spec_name.setter
    def spec_name(self, new_spec_name):
        self._spec_name = new_spec_name

    @property
    def num_troughs(self):
        return self._num_troughs
        
    @num_troughs.setter
    def num_troughs(self, new_num_troughs):
        self._num_troughs = new_num_troughs

    @property
    def ew(self):
        return self._ew
        
    @ew.setter
    def ew(self, new_ew):
        self._ew = new_ew

    @property
    def ew_err(self):
        return self._ew_err
        
    @ew_err.setter
    def ew_err(self, new_err_ew_err):
        self._ew_err = new_err_ew_err





