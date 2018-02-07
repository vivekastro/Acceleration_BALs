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
from scipy.optimize import curve_fit 
from matplotlib.backends.backend_pdf import PdfPages
from con_fit_iterative import *

def lam2vel(wavelength,ion_w):
    zlambda = (wavelength-ion_w)/ion_w
    R = 1./(1+zlambda)
    vel_ion = -const.c.to('km/s').value*(R**2-1)/(R**2+1)
    return vel_ion



def dered_flux(Av,wave,flux):
    dered_flux = np.zeros(len(wave))
    for i in range(len(wave)):
        ext_lambda= extinction.extinction_ccm89(wave[i] * U.angstrom, a_v= Av, r_v= 3.1)
        tau_lambda= ext_lambda/(1*1.086) 
        dered_flux[i]= flux[i] * np.exp(tau_lambda)
    return dered_flux

def ext_coeff(lamb):
    
    inv_lamba=[0.45,0.61,0.8,1.82,2.27,2.7,3.22,3.34,3.46,3.6,3.75,3.92,4.09,4.28,4.50,4.73,5.00,5.24,5.38,5.52,5.70,5.88,6.07,6.27,6.48,6.72,6.98,7.23,7.52,7.84]
    smc_ext=[-2.61,-2.47,-2.12,0.0,1.0,1.67,2.29,2.65,3.0,3.15,3.49,3.91,4.24,4.53,5.3,5.85,6.38,6.76,6.9,7.17,7.71,8.01,8.49,9.06,9.28,9.84,10.8,11.51,12.52,13.54]
    xy=np.interp(1.0/(lamb*10**(-4)),inv_lamba,smc_ext)
    ext_lamb=(xy+2.98)/3.98 # Rv=2.98
    #return(z,ext_lamb)
    return(ext_lamb)

def myfunct(wave, a , b):
    return a*ext_coeff(wave)*wave**(b)


def compute_alpha(wl, spec, ivar, wav_range, per_value=[10,90]):
    # print 'Routine begins'
    spec[np.isnan(spec)] = 0
    ivar[np.isnan(ivar)] = 0

    wavelength, spectra, invar = np.array([]), np.array([]), np.array([])
    #plt.plot(wl,spec)
    for j in range(len(wav_range)):
        #print wav_range[j]
        #print min(wl),max(wl)
        temp = np.where((wl > wav_range[j][0]) & (wl < wav_range[j][1]))[0]
        #print wl[temp],len(spec),len(wl),spec[temp],ivar[temp]
        tempspec, tempivar  = spec[temp], ivar[temp]
        #print tempspec
        print len(tempspec)
        #Mask out metal absorption lines
        cut = np.percentile(tempspec, per_value)
        #print 'cut',cut
        blah = np.where((tempspec > cut[0]) & (tempspec < cut[1])  & (tempivar > 0))[0]
        wave = wl[temp][blah]

        wavelength = np.concatenate((wavelength, wave))
        spectra = np.concatenate((spectra, tempspec[blah]))
        invar = np.concatenate((invar, tempivar[blah]))
    print 'Debug',len(wavelength)
    p0=[1.0,1.0]
    param_bounds = ([0,-3],[np.inf,3])
    try:
        #plt.plot(wavelength,spectra)
        #plt.show()
        popt, pcov = curve_fit(myfunct, wavelength, spectra, p0, sigma=1.0/np.sqrt(invar),bounds=param_bounds)
    except (RuntimeError, TypeError):
        AMP, ALPHA, CHISQ, DOF = np.nan, np.nan, np.nan, np.nan
    else:
        AMP, ALPHA = popt[0], popt[1]
        CHISQ = np.sum(invar * (spectra - myfunct(wavelength, popt[0], popt[1]))**2)
        # DOF = N - n  , n = 2
        DOF = len(spectra) - 2
    # print 'Routine ends' 
    print 'Compute Alpha:', AMP,ALPHA,CHISQ,DOF
    return AMP, ALPHA, CHISQ, DOF


def waveRange(wave):
    #wav_range=[[1300.,1350.],[1420.,1470.],[1700.,1800.],[2080.,2215],[2480.,2655],[3225.,3900.],[4200.,4230.],[4435.,4700.],[5200.,5700.]]
    wav_range= [(1250,1350),(1700,1800),(1950,2200),(2650,2710),(2950,3700),(3950,4050)]


    mwav_range = []
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
    return mwav_range


def powerlawFunc(xdata, scale, amp,index):
    return scale + amp*ext_coeff(xdata)*np.power(xdata,index)


def fitPowerlaw(wave,flux,weight,scale = 1, amp=1,index=1): 
    from lmfit import minimize, Parameters, fit_report, Minimizer
    import numpy as np
    import scipy.optimize as optimization
    x0= [scale, amp,index]
    xdata=np.asarray(wave)
    ydata=np.asarray(flux)
    sigma=np.asarray(weight)
    print len(xdata),len(ydata),len(sigma) 
        
    popt, pcov = optimization.curve_fit(powerlawFunc, xdata, ydata, x0, sigma)
    print popt
    #popt, pcov = optimization.curve_fit(func, xdata, ydata, x0)
    model = powerlawFunc(wave,popt[0],popt[1],popt[2])
    chi2 = ((flux - model)*np.sqrt(weight))**2
    rchi2 = np.sum(chi2)/(len(xdata) - 2)
    print 'Reduced Chi Square : {0}  Number of points: {1}'.format(rchi2,len(xdata))
    return (popt,pcov)   

def maskOutliers(wave,flux,weight,amp,alpha):
    model = myfunct(wave,amp,alpha)
    std =np.std(flux[weight > 0])
    fluxdiff = flux - model
    ww = np.where (np.abs(fluxdiff) > 3*std)
    #nwave = np.delete(wave,ww)
    #nflux = np.delete(flux,ww)
    weight[ww] = 0#np.delete(weight,ww)
    
    return wave,flux,weight

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except (ValueError, msg):
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = np.asarray(y[0]) - np.abs( y[1:half_window+1][::-1] - np.asarray(y[0]) )
    lastvals = np.asarray(y[-1]) + np.abs(y[-half_window-1:-1][::-1] - np.asarray(y[-1]))
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')





def contWave(wave):
    linefree_regions = [(1280,1350),(1700,1800),(1950,2200),(2650,2710),(2950,3700),(3950,4050)]
    finalcond = False
    for lfr in linefree_regions:
        cond = ((wave >= lfr[0]) & (wave <= lfr[1]))
        finalcond = finalcond | cond
    indices = np.where(finalcond)
    return indices


def minmax(vec):
    return min(vec),max(vec)

#version I = no maskOutliers percentile 25 75
#version II = with maskOutliers percentile 10 90
#version III = clipped wavelengths to same bounds; to deal with the difference in spectral indices
#version IV = Kate model included, ALPHA_AMP_values file is not different from III
#version V =Reddened power law constrained fit parameters to bounds -3 to 3 for alpha 0 to inf

df = np.genfromtxt('tdss_allmatches_crop_edit.dat',names=['ra','dec','z','pmf1','pmf2','pmf3'],dtype=(float,float,float,'|S15','|S15','|S15'))
wav_range= [(1280,1350),(1700,1800),(1950,2200),(2650,2710),(3010,3700),(3950,4050),(4140,4270)]
pp = PdfPages('ContinuumNormalization_plots_V.pdf')
pp1 = PdfPages('NormalizedSpectra_plots_V.pdf')
fx=open('ALPHA_AMP_values_V.txt','w')
fx1=open('Kate_ALPHA_AMP_values_V.txt','w')
#for i in range(len(df['pmf1'])):
for i in range(len(df)):
#for i in range(25):

    print 'Kate_Sources/spec-'+df['pmf1'][i]+'.fits' 
    print 'Kate_Sources/spec-'+df['pmf2'][i]+'.fits' 
    print 'Kate_Sources/spec-'+df['pmf3'][i]+'.fits' 
    data1 = fits.open('Kate_Sources/spec-'+df['pmf1'][i]+'.fits')[1].data
    data2 = fits.open('Kate_Sources/spec-'+df['pmf2'][i]+'.fits')[1].data
    data3 = fits.open('Kate_Sources/spec-'+df['pmf3'][i]+'.fits')[1].data
    wave1 = 10**data1.loglam.copy()
    flux1 = data1.flux.copy()
    sigma1 = data1.ivar.copy()
    mask1 = data1.and_mask.copy()
    weight1 =  (data1.ivar*(data1.and_mask == 0)).copy()
    wave2 = 10**data2.loglam.copy()
    flux2 = data2.flux.copy()
    sigma2 = data2.ivar.copy()
    mask2 = data2.and_mask.copy()
    weight2 =  (data2.ivar*(data2.and_mask == 0)).copy()
    wave3 = 10**data3.loglam.copy()
    flux3 = data3.flux.copy()
    sigma3 = data3.ivar.copy()
    mask3 = data3.and_mask.copy()
    weight3 =  (data3.ivar*(data3.and_mask == 0)).copy()
    print len(wave1),len(flux1),len(weight1)
    print weight1
    print data1.and_mask
    
        
    #de-redden the flux
    info = fits.open('Kate_Sources/spec-'+df['pmf1'][i]+'.fits')[2].data
    coords = SkyCoord(info['RA'],info['DEC'],unit='degree',frame='icrs')
    sfd = SFDQuery()
    eb_v = sfd(coords)
    dered_flux1 = dered_flux(3.1*eb_v,wave1,flux1)
    dered_flux2 = dered_flux(3.1*eb_v,wave2,flux2)
    dered_flux3 = dered_flux(3.1*eb_v,wave3,flux3)

    #clean QSO
    sn1= flux1*np.sqrt(weight1) ; sn2= flux2*np.sqrt(weight2); sn3= flux3*np.sqrt(weight3)
    w1 = (weight1>0)&((sn1<-10)|(sn1>80)); w2 = (weight2>0)&((sn2<-10)|(sn2>80)) ; w3 = (weight3>0)&((sn3<-10)|(sn3>80))
    print w1,w2,w3
    weight1[w1] = 0; flux1[w1] = 0
    weight2[w2] = 0; flux2[w2] = 0
    weight3[w3] = 0; flux3[w3] = 0

    
    # Change wavelengths to rest wavellegths
    rwave1 = wave1/(1.0+df['z'][i])
    rwave2 = wave2/(1.0+df['z'][i])
    rwave3 = wave3/(1.0+df['z'][i])
    
    #Clip all the spectra to same wavelength bounds Necessary to capture variation in alpha
    wmin1,wmax1 = minmax(rwave1) 
    wmin2,wmax2 = minmax(rwave2) 
    wmin3,wmax3 = minmax(rwave3)
    wlim1,wlim2 = max([wmin1,wmin2,wmin3]),min([wmax1,wmax2,wmax3])
    l1 = np.where((rwave1 >= wlim1) & (rwave1 <= wlim2))[0]
    l2 = np.where((rwave2 >= wlim1) & (rwave2 <= wlim2))[0]
    l3 = np.where((rwave3 >= wlim1) & (rwave3 <= wlim2))[0]
    crwave1 = rwave1[l1];cdered_flux1 = dered_flux1[l1];cweight1 = weight1[l1];csigma1 = sigma1[l1];cmask1 = mask1[l1]
    crwave2 = rwave2[l2];cdered_flux2 = dered_flux2[l2];cweight2 = weight2[l2];csigma2 = sigma2[l2];cmask2 = mask2[l2]
    crwave3 = rwave3[l3];cdered_flux3 = dered_flux3[l3];cweight3 = weight3[l3];csigma3 = sigma3[l3];cmask3 = mask3[l3]

    # Choose line free region
    #cwave1  = rwave1[contWave(rwave1)];cflux1 = flux1[contWave(rwave1)] ; cweight1 = weight1[contWave(rwave1)]
    #cwave2  = rwave2[contWave(rwave2)];cflux2 = flux2[contWave(rwave2)] ; cweight2 = weight2[contWave(rwave2)]
    #cwave3  = rwave3[contWave(rwave3)];cflux3 = flux3[contWave(rwave3)] ; cweight3 = weight3[contWave(rwave3)]
    
    #Fit powerlaw iterate and mask outliers
    iteration = 3
    for j in range(iteration):
        if j == 0:
            AMP1, ALPHA1, CHISQ1, DOF1 = compute_alpha(crwave1, cdered_flux1, cweight1, waveRange(crwave1))
            AMP2, ALPHA2, CHISQ2, DOF2 = compute_alpha(crwave2, cdered_flux2, cweight2, waveRange(crwave2))
            AMP3, ALPHA3, CHISQ3, DOF3 = compute_alpha(crwave3, cdered_flux3, cweight3, waveRange(crwave3))
            print AMP1,ALPHA1
            nwave1 = crwave1; nflux1=cdered_flux1;nweight1 = cweight1
            nwave2 = crwave2; nflux2=cdered_flux2;nweight2 = cweight2
            nwave3 = crwave3; nflux3=cdered_flux3;nweight3 = cweight3
            print 'iteration number',j+1#,popt1[0],popt1[1],popt1[2]
            print 'iteration 1 completed'
            continue
        else:
            nwave1,nflux1,nweight1 = maskOutliers(nwave1,nflux1,nweight1,AMP1,ALPHA1)
            AMP1, ALPHA1, CHISQ1, DOF1 = compute_alpha(nwave1, nflux1, nweight1, waveRange(nwave1))
            nwave2,nflux2,nweight2 = maskOutliers(nwave2,nflux2,nweight2,AMP2,ALPHA2)
            AMP2, ALPHA2, CHISQ2, DOF2 = compute_alpha(nwave2, nflux2, nweight2, waveRange(nwave2))
            nwave3,nflux3,nweight3 = maskOutliers(nwave3,nflux3,nweight3,AMP3,ALPHA3)
            AMP3, ALPHA3, CHISQ3, DOF3 = compute_alpha(nwave3, nflux3, nweight3, waveRange(nwave3))
           # print 'iteration number',j+1,popt1[0],popt1[1],popt1[2]
           # nwave1,nflux1,nweight1 = maskOutliers(cwave1,cflux1*1e15,cweight1,popt1)
           # popt1,pcov1 = fitPowerlaw(nwave1,nflux1,nweight1,popt1[0],popt1[1],popt1[2])
           # nwave2,nflux2,nweight2 = maskOutliers(cwave2,cflux2*1e15,cweight2,popt2)
           # popt2,pcov2 = fitPowerlaw(nwave2,nflux2,nweight2,popt2[0].popt2[1],popt2[2])
           # nwave3,nflux3,nweight3 = maskOutliers(cwave3,cflux3*1e15,cweight3,popt3)
           # popt3,pcov3 = fitPowerlaw(nwave3,nflux3,nweight3,popt3[0],popt3[1],popt3[2])
        

    print>>fx,'{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}'.format(df['pmf1'][i],ALPHA1,ALPHA2,ALPHA3,AMP1,AMP2,AMP3)
    #Try Kate's continuum normalization fit
    kcwave1,kcflux1,kcfluxerr1,kmean_amp1,kmean_index1,kredco1,kwave1,kflux1=continuum_fit(wave1,flux1,weight1,df['z'][i])
    kcwave2,kcflux2,kcfluxerr2,kmean_amp2,kmean_index2,kredco2,kwave2,kflux2=continuum_fit(wave2,flux2,weight2,df['z'][i])
    kcwave3,kcflux3,kcfluxerr3,kmean_amp3,kmean_index3,kredco3,kwave3,kflux3=continuum_fit(wave3,flux3,weight3,df['z'][i])
    print>>fx1,'{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}'.format(df['pmf1'][i],kmean_index1,kmean_index2,kmean_index3,kmean_amp1,kmean_amp2,kmean_amp3)
    #
    #Plot for testing
    fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(15,8))
    ax1.plot(crwave1,cdered_flux1,label=str(df['pmf1'][i]))
    #ax1.plot(cwave1,cflux1,'--')
    ax1.plot(crwave1,myfunct(crwave1,AMP1,ALPHA1),':',color='red',lw=3,label='my model')
    ax1.plot(kcwave1/(1.0+df['z'][i]),kcflux1,'--',color='cyan',lw=3,label='Kate model')
    #ax1.plot(rwave1,dered_flux1,'--')
    #ax1.plot(rwave1,weight1,':',label='weight')
    #ax1.plot(rwave1,data1.flux,label='Clipped')
    string1 = 'AMP: {0:4.3f}   ALPHA: {1:4.3f}   rCHISQ: {2:4.3f}'.format(AMP1,ALPHA1,CHISQ1/DOF1)
    
    for ll in wav_range :
            ax1.axvspan(ll[0], ll[1], alpha=0.25, color='cyan')
            ax2.axvspan(ll[0], ll[1], alpha=0.25, color='cyan')
            ax3.axvspan(ll[0], ll[1], alpha=0.25, color='cyan')
    ax2.plot(crwave2,cdered_flux2,label=str(df['pmf2'][i]))
    #ax2.plot(cwave2,cflux2,'--')
    ax2.plot(crwave2,myfunct(crwave2,AMP2,ALPHA2),':',color='red',lw=3,label='my model')
    ax2.plot(kcwave2/(1.0+df['z'][i]),kcflux2,'--',color='cyan',lw=3,label='Kate model')
    #ax2.plot(rwave2,dered_flux2,'--')
    #ax2.plot(rwave2,weight2,':',label='weight')
    #ax2.plot(rwave2,data2.flux,'--',label='Clipped')
    string2 = 'AMP: {0:4.3f}   ALPHA: {1:4.3f}   rCHISQ: {2:4.3f}'.format(AMP2,ALPHA2,CHISQ2/DOF2)

    ax3.plot(crwave3,cdered_flux3,label=str(df['pmf3'][i]))
    #ax3.plot(cwave3,cflux3,'--')
    ax3.plot(crwave3,myfunct(crwave3,AMP3,ALPHA3),':',color='red',lw=3,label='my model')
    ax3.plot(kcwave3/(1.0+df['z'][i]),kcflux3,'--',color='cyan',lw=3,label='Kate model')
    #ax3.plot(rwave3,dered_flux3,'--')
    #ax3.plot(rwave3,weight3,':',label='weight')
    #ax3.plot(rwave3,data3.flux,'--',label='Clipped')
    string3 = 'AMP: {0:4.3f}   ALPHA: {1:4.3f}   rCHISQ: {2:4.3f}'.format(AMP3,ALPHA3,CHISQ3/DOF3)

    ax1.set_xlim(np.min(crwave1),np.max(crwave1))
    ax2.set_xlim(np.min(crwave1),np.max(crwave1))
    ax3.set_xlim(np.min(crwave1),np.max(crwave1))
    xlim=ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ylim2 = ax2.get_ylim()
    ylim3 = ax3.get_ylim()
    ax1.text(xlim[0]+0.1*(xlim[1] - xlim[0]),ylim1[1] - 0.1*(ylim1[1] - ylim1[0]),string1)
    ax2.text(xlim[0]+0.1*(xlim[1] - xlim[0]),ylim2[1] - 0.1*(ylim2[1] - ylim2[0]),string2)
    ax3.text(xlim[0]+0.1*(xlim[1] - xlim[0]),ylim3[1] - 0.1*(ylim3[1] - ylim3[0]),string3)

    ax1.legend(loc=1)
    ax2.legend(loc=1)
    ax3.legend(loc=1)
    fig.tight_layout()
    fig.savefig(pp,format='pdf')
    #plt.show()

    #Save normalized flux for CIV region
    nl1 = np.where((crwave1 >= 1410) & (crwave1 <= 1650))[0]
    nl2 = np.where((crwave2 >= 1410) & (crwave2 <= 1650))[0]
    nl3 = np.where((crwave3 >= 1410) & (crwave3 <= 1650))[0]
    normwave1 = crwave1[nl1] ; normflux1 = cdered_flux1[nl1]/myfunct(normwave1,AMP1,ALPHA1) ; normsigma1 =  csigma1[nl1]/myfunct(normwave1,AMP1,ALPHA1) ;normmask1 = cmask1[nl1]   
    normwave2 = crwave2[nl2] ; normflux2 = cdered_flux2[nl2]/myfunct(normwave2,AMP2,ALPHA2) ; normsigma2 =  csigma2[nl2]/myfunct(normwave2,AMP2,ALPHA2) ;normmask2 = cmask2[nl2]   
    normwave3 = crwave3[nl3] ; normflux3 = cdered_flux3[nl3]/myfunct(normwave3,AMP3,ALPHA3) ; normsigma3 =  csigma3[nl3]/myfunct(normwave3,AMP3,ALPHA3) ; normmask3= cmask3[nl3]   
    outname1 = 'Normspec_'+str(df['pmf1'][i])+'.txt' 
    outname2 = 'Normspec_'+str(df['pmf2'][i])+'.txt' 
    outname3 = 'Normspec_'+str(df['pmf3'][i])+'.txt'
    np.savetxt(outname1,zip(normwave1,normflux1,normsigma1,normmask1), fmt='%10.5f')
    np.savetxt(outname2,zip(normwave2,normflux2,normsigma2,normmask2), fmt='%10.5f')
    np.savetxt(outname3,zip(normwave3,normflux3,normsigma3,normmask3), fmt='%10.5f')


    fig1,aax=plt.subplots(figsize=(15,8))
    aax.plot(normwave1,normflux1,color='black',alpha=0.4,label=str(df['pmf1'][i]))
    aax.plot(normwave2,normflux2,color='red',alpha=0.4,label=str(df['pmf2'][i]))
    aax.plot(normwave3,normflux3,color='blue',alpha=0.4,label=str(df['pmf3'][i]))
    aax.axhline(1.0,ls='--',color='orange')
    aax.set_xlabel(r'Rest Wavelength')
    aax.set_ylabel(r'Normalized Flux')
    aax.legend(loc=1)
    fig1.tight_layout()
    fig1.savefig(pp1,format='pdf')
fx.close()
fx1.close()
pp.close()
pp1.close()
