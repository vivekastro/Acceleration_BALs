#!/usr/bin/python
#-*- coding: utf-8 -*-

'''
Code for continuum normalization of SDSS spectra
@author   : Vivek M
@date     : 21-Jan-2018
@version  : 1.0
'''



import sys
import os
import numpy as np
from astropy import io
from astropy.io import fits 
from astropy.io import ascii
from matplotlib import pyplot as plt
from astropy import constants as const
from astropy import units as U
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from specutils import extinction  
import scipy
from scipy.ndimage.filters import convolve
from scipy import interpolate
from scipy.interpolate import interp1d
import scipy.optimize as optimization
from scipy import optimize
from scipy.optimize import curve_fit

def maskOutliers(wave,flux,weight,popt):
    model = powerlawFunc(wave,popt[0],popt[1],popt[2])
    std =np.std(flux[weight > 0])
    fluxdiff = flux - model
    ww = np.where (np.abs(fluxdiff) > 1*std)
    nwave = np.delete(wave,ww)
    nflux = np.delete(flux,ww)
    nweight = np.delete(weight,ww)
    
    return nwave,nflux,nweight

def contWave(wave):
    linefree_regions = [(1250,1350),(1700,1800),(1950,2200),(2650,2710),(2950,3700),(3950,4050)]
    finalcond = False
    for lfr in linefree_regions:
        cond = ((wave >= lfr[0]) & (wave <= lfr[1]))
        finalcond = finalcond | cond
    indices = np.where(finalcond)
    return indices


def powerlawFunc(xdata, amp,index):
    return  amp*xdata**index


def fitPowerlaw(wave,flux,sigma,p):
    p0 = p
    popt,pcov =  optimization.curve_fit(powerlawFunc,wave,flux,p0,sigma)
    return popt,pcov

def myfunct(wave, a , b):
    return a*wave**(b)


def compute_alpha(wl, spec, ivar, wav_range, per_value):
    # print 'Routine begins'
    wavelength, spectra, invar = np.array([]), np.array([]), np.array([])

    for j in range(len(wav_range)):
        temp = np.where((wl > wav_range[j][0]) & (wl < wav_range[j][1]))[0]
        tempspec, tempivar  = spec[temp], ivar[temp]
        #print tempspec
            
        #Mask out metal absorption lines
        cut = np.percentile(tempspec, per_value)
        #print 'cut',cut
        blah = np.where((tempspec > cut) & (tempivar > 0))[0]
        wave = wl[temp][blah]

        wavelength = np.concatenate((wavelength, wave))
        spectra = np.concatenate((spectra, tempspec[blah]))
        invar = np.concatenate((invar, tempivar[blah]))

    try:
        popt, pcov = curve_fit(myfunct, wavelength, spectra, sigma=1.0/np.sqrt(invar))
    except (RuntimeError, TypeError):
        AMP, ALPHA, CHISQ, DOF = np.nan, np.nan, np.nan, np.nan
    else:
        AMP, ALPHA = popt[0], popt[1]
        CHISQ = np.sum(invar * (spectra - myfunct(wavelength, popt[0], popt[1]))**2)
        # DOF = N - n  , n = 2
        DOF = len(spectra) - 2
    # print 'Routine ends' 

    return AMP, ALPHA, CHISQ, DOF

per_value= 0.2
wav_range=[[1300.,1350.],[1420.,1470.],[1700.,1800.],[2080.,2215],[2480.,2655],[3225.,3900.],[4200.,4230.],[4435.,4700.],[5200.,5700.]]
mwav_range = []
        
data = fits.open('Kate_Sources/spec-0389-51795-0332.fits')[1].data
flux =data.flux
wave = (10**data.loglam)/3.85
weight = (data.ivar*(data.and_mask == 0))

for rr in wav_range:
    if ((np.min(wave) < rr[0]) & (np.max(wave) > rr[1])):
        nrr0 = rr[0] ; nrr1 = rr[1]
    elif ((np.min(wave) > rr[0]) & (np.min(wave) < rr[1])& (np.max(wave) < rr[1])):
        nrr0 = np.min(wave) ; nrr1 = rr[1]
    elif ((np.min(wave) < rr[0]) & (np.max(wave) > rr[0]) &(np.max(wave) < rr[1])):
        nrr0 = rr[0] ; nrr1 = np.max(wave)
    else :
        continue
    mwav_range.append([nrr0,nrr1])
        
        #print mwav_range


AMP, ALPHA, CHISQ, DOF = compute_alpha(wave, flux, weight, mwav_range, per_value)
cwave = wave[contWave(wave)]
cflux = flux[contWave(wave)]
cweight = weight[contWave(wave)]
#sigma = cweight
#p0=[1,1]
#popt,pcov =  optimization.curve_fit(powerlawFunc,cwave,cflux,p0,sigma)
#print popt
plt.plot(cwave,cflux)
plt.plot(wave,AMP*wave**ALPHA,'--')
plt.show()
 
