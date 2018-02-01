#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
import argparse
import os
import numpy as np
from astropy import io
from astropy.io import fits as pyfits
from astropy.io import ascii
from astropy.table import Table
from matplotlib import pyplot as plt
import scipy
from scipy.ndimage.filters import convolve
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy import optimize 
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
#import crop_spectrum #not used in this version 
from Observation import Observation
import CCF_interp as myccf
from scipy import stats 

#Wavelengths of each species
nv_0 = 1238.821
siiv_0 = 1393.755
civ_0 = 1548.202
aliii_0 = 1854.7164
mgii_0 = 2796.352
lightspeed = 2.99792458e5
nu_0 = civ_0

#crop = True #(False if I've already done cropping and just want to redo the lags). This is from an old version; not used here. 

#Convert wavelength to velocity
def lam2vel(wavelength):
    zlambda = (wavelength-ion_0)/ion_0
    R = 1./(1+zlambda)
    vel_ion = -lightspeed*(R**2-1)/(R**2+1)
    return vel_ion

###################################################################################################
## Directories and list of files to use
###################################################################################################
tdssdatadir = '../../varbal_data/all_textfiles/'
bossdatadir = '../../varbal_data/all_textfiles/'
sdssdatadir = '../../varbal_data/all_textfiles/'

#bi_file = '/Users/cjg235/balqsos/varbal/measurements/BI_out_multitrough.dat'
bi_file = '../../measurements/pyccf_tries_69kms_feb/trough_complexes.dat'
datafile = '../../sample_selection/tdss_allmatches_crop_edit.dat' 

#Read in file containing all plate-mjd-fibers of pairs of spectra I want to use 
targ_obs = list()
with open(datafile) as file:
    for line in file:
        if line.startswith('#'):
            continue
        else:
            targ_obs.append(line.rstrip("\n"))

num_obj = np.size(targ_obs)
print 'Number of different targets: ', num_obj

target_observations = [[] for _ in range(num_obj)]
print target_observations
exit()

#bi_info = np.genfromtxt(bi_file, dtype = None, names = ['files', 'num_troughs', 'BI', 'BI_err', 'BIround', 'Vmax', 'Verr', 'Vmaxround', 'Vmin', 'Verr2', 'Vminround', 'Chi2'], skiprows = 1)
#num_bi = np.size(bi_info['num_troughs'])
bi_info = np.genfromtxt(bi_file, dtype = None, names = ['speci', 'files', 'Vmin', 'Vmax', 'numt', 'num_troughs', 'highpad', 'lowpad', 'ew', 'ew_err'], skiprows = 1)
num_bi = np.size(bi_info['num_troughs'])
x = np.genfromtxt('../../data_preparation/line_fits.dat', dtype = None, names = ['numobj', 'spec1', 'spec2', 'spec3', 'linefit'], skiprows = 1)
linefit = x['linefit']
               
for i in xrange(0, num_obj):
    values = targ_obs[i].split()
    ra = values[0]
    dec = values[1]
    redshift = values[2]
    valuesn = sorted(values[3:])
    valuesize = np.size(valuesn)
    line_fit = linefit[i]
    if line_fit == 'gauss-hermite':
        file_suffix = '-gh-norm.txt'
    elif line_fit == 'voigt':
        file_suffix = '-voigt-norm.txt'
    else:
        file_suffix = '-2g-norm.txt'
    
    #print valuesize, valuesn
    for j in xrange(0, valuesize):
        obs = Observation() 
        platemjdfiber = valuesn[j]
        plate = platemjdfiber.split('-')[0] 
        mjd = int(platemjdfiber.split('-')[1])
        fiber = platemjdfiber.split('-')[2]
        if mjd < 55050:
            #print 'sdss'
            filename = sdssdatadir+'spec-'+valuesn[j]+file_suffix
            survey = 'SDSS'
        if np.logical_and(mjd >= 55100., mjd <= 56850.):
            #print 'boss'
            filename = bossdatadir+'spec-'+valuesn[j]+file_suffix        
            survey = 'BOSS'
        if mjd > 56850.:
            #print 'tdss'
            filename = tdssdatadir+'spec-'+valuesn[j]+file_suffix   
            survey = 'TDSS'

        spec_filename = 'spec-'+valuesn[j]+file_suffix
        match = np.where(bi_info['files'] == spec_filename)[0]

        if np.size(match) == 0:
            obs.filename = filename
            obs.spec_name = spec_filename
            obs.mjd = mjd
            obs.pmf = platemjdfiber
            obs.survey = survey
            obs.redshift = redshift 
            obs.num_troughs = [0]
            obs.v_max = 0
            obs.v_min = 0
            print i, spec_filename
        else:
            obs.filename = filename
            obs.spec_name = spec_filename
            obs.mjd = mjd
            obs.pmf = platemjdfiber
            obs.survey = survey
            obs.redshift = redshift 
            obs.num_troughs = bi_info['num_troughs'][match]
            obs.v_max = bi_info['Vmax'][match]
            obs.v_min = bi_info['Vmin'][match]

        target_observations[i].append(obs)


print 'Carrying out CCF analysis' 

#Output files
lagsout = open('shift_measurements.dat', 'w')
lagsout.write('#Plot, Target-i, Spec1       Spec2       delta_t (years)  Centroid       UpErr     Lowerr       Peak       UpErr     Lowerr       R_val     Accel (cm/s^2)    UpErr     Lowerr       Shift      UpErr     Lowerr     vmin vmax  \n') 
lagsout.close()
lagsout3 = open('shift_measurements_3sigma.dat', 'w')
lagsout3.write('#Plot, Target-i, Spec1       Spec2       delta_t (years)  Centroid       UpErr     Lowerr       Peak       UpErr     Lowerr       R_val     Accel (cm/s^2)    UpErr     Lowerr       Shift      UpErr     Lowerr     vmin vmax  \n') 
lagsout3.close()

pdf_pages = PdfPages('all_ccfs.pdf')
numplot = 0
for i in xrange(0, num_obj):   #num_obj
    target = target_observations[i]
    numrep = np.size(target)
    n_troughs = target[0].num_troughs[0]
    if n_troughs > 0: 
        for z in xrange(0, n_troughs):
            suffix = str(z)
            for j in xrange(0, numrep-1):
                k = 0
                while (j+k) < (numrep-1):
                    k+=1

                    print target[j].filename
                    spec1 = 'fivecol/'+(target[j].filename).split('/')[7]+'.crop.5col_'+suffix
                    spec2 = 'fivecol/'+(target[j+k].filename).split('/')[7]+'.crop.5col_'+suffix
                    prefix1 = ((target[j].filename).split('/')[7]).split('-')[1]+'-'+ ((target[j].filename).split('/')[7]).split('-')[2]+'-'+ ((target[j].filename).split('/')[7]).split('-')[3]
                    prefix2 = ((target[j+k].filename).split('/')[7]).split('-')[1]+'-'+ ((target[j+k].filename).split('/')[7]).split('-')[2]+'-'+ ((target[j+k].filename).split('/')[7]).split('-')[3]
                    prefix = prefix1+'-vs-'+prefix2+'-'+suffix
                    print 'i = ', i, ' Cross correlating: ', prefix
                    deltamjd = abs(int(target[j+k].mjd) - int(target[j].mjd))
                    deltamjd_rest = (deltamjd/(1+float(target[j].redshift)))/365.
            
                    #Calculate lag with python CCF program
                    nsim = 10000
                    wave1, flux1, err1 = np.loadtxt(spec1, unpack = True, usecols = [0, 1, 2])
                    wave2, flux2, err2 = np.loadtxt(spec2, unpack = True, usecols = [0, 1, 2])
                    #print z
                    #print target[j].v_min
                    vmin = target[j].v_min[z]
                    vmax = target[j].v_max[z] 
                
                    #Run the CCF on the two spectra
                    #Run the CCF from -2000 km/s to +2000 km/s and at 69 km/s interpolation
                    tlag_peak, status_peak, tlag_centroid, status_centroid, ccf_pack, max_r, status_r = myccf.peakcent(-wave1, flux1, -wave2, flux2, -2.001, 2.001, 0.069, imode = 0, sigmode = 0.2)
                    tlags_peak, tlags_centroid, nsuccess_peak, nfail_peak, nsuccess_centroid, nfail_centroid, max_rvals, nfail_maxrvals = myccf.xcor_mc(-wave1, flux1, abs(err1), -wave2, flux2, abs(err2), -2.001, 2.001, 0.069, nsim = nsim, mcmode=2, sigmode = 0.2)
                    #plt.plot(ccf_pack[1], ccf_pack[0])
                    #plt.show()
                    #plt.hist(tlags_centroid, color = 'b')
                    #plt.hist(tlags_peak, color = 'r')
                    #plt.show()
                    
                    #Write out to file
                    centfile = open('ccf_files/centdist_'+prefix, 'w')
                    peakfile = open('ccf_files/peakdist_'+prefix, 'w')
                    ccf_file = open('ccf_files/ccf_'+prefix, 'w')
                    lag = ccf_pack[1]
                    r = ccf_pack[0]
                    for m in xrange(0, np.size(tlags_centroid)):
                        centfile.write('%5.5f \n'%(tlags_centroid[m]))
                    for m in xrange(0, np.size(tlags_peak)):
                        peakfile.write('%5.5f \n'%(tlags_peak[m]))
                    centfile.close()
                    peakfile.close()
                    for m in xrange(0, np.size(lag)):
                        ccf_file.write('%5.5f    %5.5f  \n'%(lag[m], r[m]))
                    ccf_file.close() 

                                #Calculate lag with python CCF program
                    centfile = 'ccf_files/centdist_'+prefix
                    peakfile = 'ccf_files/peakdist_'+prefix
                    ccf_file = 'ccf_files/ccf_'+prefix
                    lag, r = np.loadtxt(ccf_file, unpack = True)
                    tlags_centroid = np.loadtxt(centfile, unpack = True)
                    tlags_peak = np.loadtxt(peakfile, unpack = True) 

                    perclim = 84.1344746
                    perclim3sig = 99.8650102
                    
                    #For non-Gaussian Distribution
                    centau = stats.scoreatpercentile(tlags_centroid, 50)
                    centau_uperr = (stats.scoreatpercentile(tlags_centroid, perclim))-centau
                    centau_loerr = centau-(stats.scoreatpercentile(tlags_centroid, (100.-perclim)))
                    centau_uperr3 = (stats.scoreatpercentile(tlags_centroid, perclim3sig))-centau
                    centau_loerr3 = centau-(stats.scoreatpercentile(tlags_centroid, (100.-perclim3sig)))
                    print 'Centroid, error: ', centau, centau_loerr, centau_uperr
                    
                    peaktau = stats.scoreatpercentile(tlags_peak, 50)
                    peaktau_uperr = (stats.scoreatpercentile(tlags_peak, perclim))-centau
                    peaktau_loerr = centau-(stats.scoreatpercentile(tlags_peak, (100.-perclim)))
                    peaktau_uperr3 = (stats.scoreatpercentile(tlags_peak, perclim3sig))-centau
                    peaktau_loerr3 = centau-(stats.scoreatpercentile(tlags_peak, (100.-perclim3sig)))
                    print 'Peak, errors: ', peaktau, peaktau_uperr, peaktau_loerr

                    #Convert the measured shift in velocity space to acceleration.
                    rval = np.max(r) 
                    accel_meas = (centau*1000/deltamjd_rest)*(1.0e5)/(365.*24.*60.*60.)   #in cm/s/s
                    accel_uperr = (centau_uperr*1000./deltamjd_rest)*(1.0e5)/(365.*24.*60.*60.)
                    accel_loerr = (centau_loerr*1000./deltamjd_rest)*(1.0e5)/(365.*24.*60.*60.)
                    accel_uperr3 = (centau_uperr3*1000./deltamjd_rest)*(1.0e5)/(365.*24.*60.*60.)
                    accel_loerr3 = (centau_loerr3*1000./deltamjd_rest)*(1.0e5)/(365.*24.*60.*60.)

                    shift = centau
                    shift_uperr = centau_uperr
                    shift_loerr = centau_loerr
                    shift_uperr3 = centau_uperr3
                    shift_loerr3 = centau_loerr3

                    print 'Shift Error ratio: ', shift_uperr/shift_loerr
                    print 'Accel Error ratio: ', accel_uperr/accel_loerr
                    print 'Shift3 Error ratio: ', shift_uperr3/shift_loerr3
                    print 'Accel3 Error ratio: ', accel_uperr3/accel_loerr3
                    
                    #write all of this out to a file.
                    lagsout = open('shift_measurements.dat', 'a')
                    lagsout3 = open('shift_measurements_3sigma.dat', 'a')
                    lagsout.write('%i     %i     %s     %s     %5.5f     %5.5f     %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f  \n'%(numplot, \
                                i, spec1,spec2, deltamjd_rest, centau, centau_uperr, centau_loerr, peaktau, peaktau_uperr, peaktau_loerr, rval, accel_meas, accel_uperr, accel_loerr, shift, shift_uperr, shift_loerr, vmin/1000., vmax/1000.))
                    lagsout.close() 
                    lagsout3.write('%i     %i     %s     %s     %5.5f     %5.5f     %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f      %5.5f  \n'%(numplot, \
                                i, spec1,spec2, deltamjd_rest, centau, centau_uperr3, centau_loerr3, peaktau, peaktau_uperr3, peaktau_loerr3, rval, accel_meas, accel_uperr3, accel_loerr3, shift, shift_uperr3, shift_loerr3, vmin/1000., vmax/1000.))
                    lagsout3.close() 

                    #Make plots of the spectra, CCF, CCCD, and CCPD, etc 
                    spec1w, spec1f, spec1er = np.loadtxt(spec1, unpack = True, usecols = [0, 1, 2])
                    spec2w, spec2f, spec2er = np.loadtxt(spec2, unpack = True, usecols = [0, 1, 2])
                    spec2wn = spec2w-shift

                    fig = plt.figure()
                    fig.subplots_adjust(hspace=0.2, wspace = 0.1)
            
                    ax1 = fig.add_subplot(2, 1, 1)
                    ax1.plot(spec1w, spec1f, color = 'k', label = 'First epoch')
                    ax1.plot(spec2w, spec2f, color = 'r', label = 'Second epoch')
                    try:
                        ax1.plot(spec2wn, spec2f, color = 'b', linestyle = '--', label = 'Shifted')
                    except:
                        print 'No shift measured!'
                    ax1.legend(loc = 'lower left', fontsize = 8)
                    ax1.set_title('Spectrum i = %i, Trough %s/%i, $\Delta$t$_{\\rm rest}$ = %5.3f years'%(i, suffix, n_troughs-1, deltamjd_rest), fontsize = 20)
                    ax1.text(0.05, 0.9, prefix, fontsize = 15, transform = ax1.transAxes)
                    ax1.set_xlim(30, -2)
                    ax1.set_ylabel('Normalized Flux')
                    ax1.set_xlabel('Velocity (10$^3$) km/s)')
            
                    xmin, xmax = -2,2
                    ax2 = fig.add_subplot(2, 3, 4)
                    ax2.set_ylabel('Correlation coefficient')
                    ax2.text(0.2, 0.85, 'CCF ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax2.transAxes, fontsize = 16)
                    ax2.set_xlim(xmin, xmax)
                    ax2.set_ylim(0, 1.0)
                    try:
                        ax2.plot(lag, r, color = 'k')
                    except:
                        print 'No CCF data'
                
                    ax3 = fig.add_subplot(2, 3, 5, sharex = ax2)
                    #ax3.set_ylabel('N')
                    ax3.set_xlim(xmin, xmax)
                    ax3.axes.get_yaxis().set_ticks([])
                    ax3.set_xlabel('Shift: %5.1f + %5.1f - %5.1f km/s, Accel: %5.3f+ %5.3f - %5.3f cm s$^{-2}$'%(shift*1000, shift_uperr*1000, shift_loerr*1000, accel_meas, accel_uperr, accel_loerr), fontsize = 15) 
                    ax3.text(0.2, 0.85, 'CCCD ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax3.transAxes, fontsize = 16)
                    try:
                        ax3.hist(tlags_centroid, color = 'k')
                    except:
                        print 'No centroid data'
                
                    ax4 = fig.add_subplot(2, 3, 6, sharex = ax2)
                    ax4.set_ylabel('N')
                    ax4.yaxis.tick_right()
                    ax4.yaxis.set_label_position('right') 
                    #ax4.set_xlabel('Lag (\AA)')
                    ax4.set_xlim(xmin, xmax)
                    ax4.text(0.2, 0.85, 'CCPD ', horizontalalignment = 'center', verticalalignment = 'center', transform = ax4.transAxes, fontsize = 16)
                    try:
                        ax4.hist(tlags_peak, color = 'k')
                    except:
                        print 'No peak data'
                
                    pdf_pages.savefig(fig)
                    plt.savefig('ccf_plots/ccfplot_'+prefix+'.png', format = 'png', bbox_inches = 'tight')
                    
                    plt.close(fig)
            
pdf_pages.close()
