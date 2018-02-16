#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
import numpy as np
from astropy import io
from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy.table import Table
from matplotlib import pyplot as plt
from Spectrum import Spectrum 
import scipy
from scipy.ndimage.filters import convolve
from scipy import interpolate
from scipy.interpolate import interp1d
from pylab import *
from scipy import optimize 

def runningMeanFast(x,N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

def SmoothBoxCar(x,N):
    boxcar=np.ones(N)
    return convolve(x, boxcar/boxcar.sum())

#Define the continuum power law function
#reddening is E(B - V)

def redpowerlaw(x, amp, index, reddening):
  w = x/10000  #convert wavelengths to microns
  a = [185, 27, 0.005, 0.010, 0.012, 0.030]
  l = [0.042, 0.08, 0.22, 9.7, 18, 25]
  b = [90, 5.50, -1.95, -1.95, -1.80, 0]
  n = [2, 4, 2, 2, 2, 2]
  K = [2.89, 0.91, 0.02, 1.55, 1.72, 1.89]
  R_V = 2.93
  
  xi = 0
  for i in xrange(0,6):
      xi+= a[i] / (np.power((w/l[i]), n[i]) + np.power((l[i]/w), n[i]) + b[i])

  ElB = (xi*(1+R_V)-R_V)*reddening  ##E(lambda - V)
  extinction = np.power(10, -0.4*ElB)    ##10^(-0.4*E(lambda - V))

  return (amp*(x/2000.)**(index))*np.power(10, (-0.4*(xi*(1+R_V)-R_V)*reddening))

####Write a fitting program ################
def fit_three_params(x, y, errs):

    fittingx = x
    fittingy = y
    weights = errs
    #print x[0:10], y[0:10], weights[0:10]
    p_init = [8.,  -1.0,  0.05]
       
    out, var_matrix = optimize.curve_fit(redpowerlaw, fittingx, fittingy, p_init, maxfev = 10000)
    #print out
    #print var_matrix
    pfin = out
    #cov = out[1]
    redco = pfin[2] 
    index = pfin[1]
    amp = pfin[0]
    #print amp, index, redco
    #errors = np.sqrt(np.diag(cov))
    #indexErr = errors[0]
    #ampErr = errors[1]
    #redcoErr = errors[2]
    modelval = redpowerlaw(fittingx, amp, index, redco)
   
    return amp, index, redco

def continuum_fit(wave, flux, error, redshift): ###########################
    spectrum = Spectrum()
    spectrum.wavelengths = wave
    spectrum.flux = flux
    spectrum.flux_error = error
    
    smooth=False
    N_kern=5 #pxl
    plotfit=True 
    fitcon=True 

    ############################################
    ###########################################
    ####### Define line-free regions and calculate weighting scheme. This will need to 
    ####### be updated to deal with different wavelength regions. 
    maxwav = np.max(spectrum.wavelengths)
    minwav = np.min(spectrum.wavelengths)

    #RLF regions from Filiz ak et al., updated a bit, and also using a different one for low-z targets. 
    if np.logical_and(redshift > 1.65, redshift < 1.85):
        index1=np.where(np.logical_and(spectrum.wavelengths>=1280.,spectrum.wavelengths<=1350.))[0]
        index2=np.where(np.logical_and(spectrum.wavelengths>=1425.,spectrum.wavelengths<=1450.))[0]
        index3=np.where(np.logical_and(spectrum.wavelengths>=1700.,spectrum.wavelengths<=1800.))[0]
        index4=np.where(np.logical_and(spectrum.wavelengths>=1950.,spectrum.wavelengths<=2200.))[0]
        index5=np.where(np.logical_and(spectrum.wavelengths>=2650.,spectrum.wavelengths<=2710.))[0]
        index6=np.where(np.logical_and(spectrum.wavelengths>=3010.,spectrum.wavelengths<=3700.))[0]
        index7=np.where(np.logical_and(spectrum.wavelengths>=3950.,spectrum.wavelengths<=4050.))[0]
        index8=np.where(np.logical_and(spectrum.wavelengths>=4140.,spectrum.wavelengths<=4270.))[0]
        index9=np.where(np.logical_and(spectrum.wavelengths>=4400.,spectrum.wavelengths<=4770.))[0]
        index10=np.where(np.logical_and(spectrum.wavelengths>=5100.,spectrum.wavelengths<=6400.))[0]
    if np.logical_or(redshift >= 1.85, redshift <= 1.65):
        index1=np.where(np.logical_and(spectrum.wavelengths>=1280.,spectrum.wavelengths<=1350.))[0]
        index2=np.where(np.logical_and(spectrum.wavelengths>=1700.,spectrum.wavelengths<=1800.))[0]
        index3=np.where(np.logical_and(spectrum.wavelengths>=1950.,spectrum.wavelengths<=2200.))[0]
        index4=np.where(np.logical_and(spectrum.wavelengths>=2650.,spectrum.wavelengths<=2710.))[0]
        index5=np.where(np.logical_and(spectrum.wavelengths>=3010.,spectrum.wavelengths<=3700.))[0]
        index6=np.where(np.logical_and(spectrum.wavelengths>=3950.,spectrum.wavelengths<=4050.))[0]
        index7=np.where(np.logical_and(spectrum.wavelengths>=4140.,spectrum.wavelengths<=4270.))[0]
        index8=np.where(np.logical_and(spectrum.wavelengths>=4400.,spectrum.wavelengths<=4770.))[0]
        index9=np.where(np.logical_and(spectrum.wavelengths>=5100.,spectrum.wavelengths<=6400.))[0]
        index10=np.where(spectrum.wavelengths>=6900.)[0]
        #print 'made it to here!'
    indexes=np.concatenate((index1, index2, index3, index4, index5, index6, index7, index8, index9, index10))
    #Calculate weights for each pixel 
    n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = np.size(index1), np.size(index2), np.size(index3), \
        np.size(index4), np.size(index5), np.size(index6), np.size(index7), np.size(index8), np.size(index9), np.size(index10) 
    ntot = n1+n2+n3+n4+n5+n6+n7+n8+n9+n10
    w1 = list() 
    w2 = list() 
    w3 = list() 
    w4 = list() 
    w5 = list() 
    w6 = list()
    w7 = list()
    w8 = list()
    w9 = list()
    w10 = list() 
    for j in xrange(0, n1):
        w1.append(ntot/float(n1))
    for j in xrange(0, n2):
        w2.append(ntot/float(n2))
    for j in xrange(0, n3):
        w3.append(ntot/float(n3))
    for j in xrange(0, n4):
         w4.append(ntot/float(n4))
    for j in xrange(0, n5):
        w5.append(ntot/float(n5))
    for j in xrange(0, n6):
        w6.append(ntot/float(n6))
    for j in xrange(0, n7):
        w7.append(ntot/float(n7))
    for j in xrange(0, n8):
        w8.append(ntot/float(n8))
    for j in xrange(0, n9):
        w9.append(ntot/float(n9))
    for j in xrange(0, n10):
        w10.append(ntot/float(n10))

    allweights = concatenate((w1, w2, w3, w4, w5, w6, w7, w8, w9, w10))
    ############################################
    ###########################################
    ##Fit the spectrum to get the actual fit first. 
    spec_to_fit1 = Spectrum()
    spec_to_fit1.wavelengths = spectrum.wavelengths[indexes]
    spec_to_fit1.flux =spectrum.flux[indexes]
    spec_to_fit1.flux_error = spectrum.flux_error[indexes]
    num_wavs_fit = np.size(spec_to_fit1.wavelengths)

    toweight = True
    if toweight: 
        weights_specfit = allweights
    else:
        weights_specfit = spec_to_fit1.flux_error
        #weights_specfit = ones(num_wavs_fit) 
    
    #Iteratively fit continuum and drop 3sigma outliers with 10 iterations. 
    newregions = spec_to_fit1.wavelengths
    newflux = spec_to_fit1.flux
    newweights = weights_specfit
    newfluxerr = spec_to_fit1.flux_error
    checksize = 10 
    while checksize > 0:
        amp, index, redco = fit_three_params(newregions, newflux, newweights)
        first_continuum_fit = redpowerlaw(spectrum.wavelengths, amp, index, redco)

    ###iterate and remove pixels that deviate from too 
        con_fit_fitreg = redpowerlaw(newregions, amp, index, redco)
        diff = con_fit_fitreg - newflux
        sigma = np.std(diff)
        toolarge = np.where(diff >= 3*sigma)
        justright = np.where(diff <= 3*sigma)
        #print np.size(newregions)
        #print np.size(toolarge)
        checksize = np.size(toolarge) 

        newregions = newregions[justright]
        newflux = newflux[justright] 
        newweights = newweights[justright]
        newfluxerr = newfluxerr[justright] 
   
    finalspecfitwave, finalspecfitflux, finalweights_specfit = newregions, newflux, newweights

    #final continuum fit
    amp, index, redco = fit_three_params(finalspecfitwave, finalspecfitflux, finalweights_specfit)
    continuum_fit = redpowerlaw(spectrum.wavelengths, amp, index, redco)

    #print 'Amplitude, Index, Reddening', amp, index, redco 
    spec_to_fit = Spectrum()
    spec_to_fit.wavelengths = finalspecfitwave
    spec_to_fit.flux = finalspecfitflux
    spec_to_fit.flux_error = newfluxerr 
    num_wavs_fit = np.size(spec_to_fit.wavelengths)
            
    toweight = True
    if toweight: 
        weights_specfit = allweights
    else:
        weights_specfit = spec_to_fit.flux_error
        #weights_specfit = ones(num_wavs_fit) 
    
    ############################################
    ###########################################
    ##### Now do Monte Carlo simulations to get the uncertainties. 
    num_samples = 100
    num_wavelengths_fit = np.size(spec_to_fit.wavelengths)
    num_wavelengths_tot = np.size(spectrum.wavelengths)
            
    if toweight:
        weights = allweights
    else:
        #weights = ones(num_wavelengths_fit)
        weights = ones(num_wavelengths_fit)
            
    consamples = zeros((num_samples, num_wavelengths_tot))
    ampsamples = zeros(num_samples)
    indexsamples = zeros(num_samples)
        
    meancon = zeros(num_wavelengths_tot) 
    errcon = zeros(num_wavelengths_tot)
      
    #Monte carlo iterations for uncertainties
    for j in xrange(0, num_samples):
        specfit = zeros(num_wavelengths_fit)
        for k in xrange(0, num_wavelengths_fit):
            specfit[k] = spec_to_fit.flux[k]+randn()*spec_to_fit.flux_error[k]

        plot2 = False
        if plot2:
            figure2, ax3 = plt.subplots(1, figsize=(12,4))
            ax3.errorbar(spectrum.wavelengths, spectrum.flux, yerr = spectrum.flux_error, color = 'b', label = 'Spec')
            ax3.plot(spectrum.wavelengths, spectrum.flux, color = 'k', label = 'Spec')
            ax3.plot(spec_to_fit.wavelengths, specfit, color = 'r', label = 'Fitting flux')
            ymin = min(0, np.percentile(spectrum.flux,  1))
            ymax = 1.25*np.percentile(spectrum.flux, 99)
            ax3.set_xlim(min(spectrum.wavelengths), max(spectrum.wavelengths))
            ax3.set_ylim(ymin, ymax)
            ax3.set_xlabel('Rest-frame Wavelength (\AA)', fontsize = 20)
            ax3.set_ylabel('Flux $(10^{-17} erg/cm^2/s/\AA)$', fontsize = 20)
            #ax2.set_title('SDSS '+str(object_info['name'][i])+ ', $z$ = '+str(object_info['z'][i]), fontsize = 22)
            ax3.tick_params(axis='x', labelsize=19)
            ax3.tick_params(axis='y', labelsize=19)
            ax3.legend()
            ax3.grid()
            plt.show()
            plt.close(figure2)

        ampout, indexout, redout = fit_three_params(spec_to_fit.wavelengths, specfit, weights)
        contflux = redpowerlaw(spectrum.wavelengths, ampout, indexout, redout)
        ampsamples[j] = ampout
        indexsamples[j] = indexout 
        for m in xrange(0, num_wavelengths_tot):
            consamples[j,m] += contflux[m]
    for j in xrange(0, num_wavelengths_tot):
        meancon[j] = mean(consamples[:,j])
        errcon[j] = std(consamples[:,j])
        
    conflux = continuum_fit
    confluxerr = errcon 
    mean_amp = amp
    err_amp = np.std(ampsamples)
    mean_index = index
    err_index = np.std(indexsamples)

    plotfit = False
    if plotfit:
        figure, ax2 = plt.subplots(1, figsize=(12,4))
        ax2.plot(spectrum.wavelengths, spectrum.flux, color = 'k', label = 'Spec')
        ax2.plot(spectrum.wavelengths, conflux, color = 'b', label = 'Continuum', linewidth = 2)
        ymin = min(0, np.percentile(spectrum.flux,  1))
        ymax = 1.25*np.percentile(spectrum.flux, 99)
        ax2.set_xlim(min(spectrum.wavelengths), max(spectrum.wavelengths))
        ax2.set_ylim(ymin, ymax)
        ax2.set_xlabel('Rest-frame Wavelength (\AA)', fontsize = 20)
        ax2.set_ylabel('Flux $(10^{-17} erg/cm^2/s/\AA)$', fontsize = 20)
        #ax2.set_title('SDSS '+str(object_info['name'][i])+ ', $z$ = '+str(object_info['z'][i]), fontsize = 22)
        ax2.tick_params(axis='x', labelsize=19)
        ax2.tick_params(axis='y', labelsize=19)
        ax2.legend()
        ax2.grid()
        plt.show()

    #Return the continuum fit, flux, uncertainty associated with the Monte Carlo iterations, and fitting regions. 
    return spectrum.wavelengths, conflux, confluxerr, mean_amp, mean_index, redco, spec_to_fit.wavelengths, spec_to_fit.flux
    
