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

import scipy
from scipy.ndimage.filters import convolve
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy import optimize 



def lam2vel(wavelength,ion_w):
    zlambda = (wavelength-ion_w)/ion_w
    R = 1./(1+zlambda)
    vel_ion = -const.c.to('km/s').value*(R**2-1)/(R**2+1)
    return vel_ion



def dered_flux(Av,wave,flux):
    ext_lambda= extinction.extinction_ccm89(wave * U.angstrom, a_v= Av, r_v= 3.1)
    tau_lambda= ext_lambda/(1*1.086) 
    dered_flux= flux * np.exp(tau_lambda)
    return (dered_flux)

def ext_coeff(lamb):
    
    inv_lamba=[0.45,0.61,0.8,1.82,2.27,2.7,3.22,3.34,3.46,3.6,3.75,3.92,4.09,4.28,4.50,4.73,5.00,5.24,5.38,5.52,5.70,5.88,6.07,6.27,6.48,6.72,6.98,7.23,7.52,7.84]
    smc_ext=[-2.61,-2.47,-2.12,0.0,1.0,1.67,2.29,2.65,3.0,3.15,3.49,3.91,4.24,4.53,5.3,5.85,6.38,6.76,6.9,7.17,7.71,8.01,8.49,9.06,9.28,9.84,10.8,11.51,12.52,13.54]
    xy=np.interp(1.0/(lamb*10**(-4)),inv_lamba,smc_ext)
    ext_lamb=(xy+2.98)/3.98 # Rv=2.98
    #return(z,ext_lamb)
    return(ext_lamb)


def pow_fita(red_datum,amp,index): 
    from lmfit import minimize, Parameters, fit_report, Minimizer
    import numpy as np
    import scipy.optimize as optimization

    x=[x[0] for x in red_datum]
    data=[x[1] for x in red_datum]
    eps_data=[x[2] for x in red_datum]
    wt=[x[3] for x in red_datum]
    
    x0=[amp,index]
    xdata=np.asarray(x)
    ydata=np.asarray(data)
    sigma=np.asarray(eps_data)
    sigma=np.multiply(wt,sigma)
    
    def func(xdata, amp, index):
        return amp*ext_coeff(xdata)*np.power(xdata,index)
    
    popt, pcov = optimization.curve_fit(func, xdata, ydata, x0, sigma)
    #popt, pcov = optimization.curve_fit(func, xdata, ydata, x0)
    
    return (popt,pcov)   



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


df = np.genfromtxt('tdss_allmatches_crop_edit.dat',names=['ra','dec','z','pmf1','pmf2','pmf3'],dtype=(float,float,float,'|S15','|S15','|S15'))

#for i in range(len(df['pmf1'])):
for i in range(2):
    print 'Kate_Sources/spec-'+df['pmf1'][i]+'.fits' 
    print 'Kate_Sources/spec-'+df['pmf2'][i]+'.fits' 
    print 'Kate_Sources/spec-'+df['pmf3'][i]+'.fits' 
    data1 = fits.open('Kate_Sources/spec-'+df['pmf1'][i]+'.fits')[1].data
    data2 = fits.open('Kate_Sources/spec-'+df['pmf2'][i]+'.fits')[1].data
    data3 = fits.open('Kate_Sources/spec-'+df['pmf3'][i]+'.fits')[1].data
    loglam1 = data1.loglam.copy()
    flux1 = data1.flux.copy()
    weight1 =  (data1.ivar*(data1.and_mask == 0)).copy()
    loglam2 = data2.loglam.copy()
    flux2 = data2.flux.copy()
    weight2 =  (data2.ivar*(data2.and_mask == 0)).copy()
    loglam3 = data3.loglam.copy()
    flux3 = data3.flux.copy()
    weight3 =  (data3.ivar*(data3.and_mask == 0)).copy()
    print len(loglam1),len(flux1),len(weight1)
    print weight1
    print data1.and_mask
    #clean QSO
    sn1= flux1*np.sqrt(weight1) ; sn2= flux2*np.sqrt(weight2); sn3= flux3*np.sqrt(weight3)
    w1 = (weight1>0)&((sn1<-10)|(sn1>80)); w2 = (weight2>0)&((sn2<-10)|(sn2>80)) ; w3 = (weight3>0)&((sn3<-10)|(sn3>80))
    print w1,w2,w3
    weight1[w1] = 0; flux1[w1] = 0
    weight2[w2] = 0; flux2[w2] = 0
    weight3[w3] = 0; flux3[w3] = 0

    fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(15,8))
    ax1.plot(10**loglam1,flux1,label=str(df['pmf1'][i]))
    ax1.plot(10**loglam1,weight1,':',label='weight')
    ax1.plot(10**loglam1,data1.flux,label='Clipped')
    ax2.plot(10**loglam2,flux2,label=str(df['pmf2'][i]))
    ax2.plot(10**loglam2,weight2,':',label='weight')
    ax2.plot(10**loglam2,data2.flux,'--',label='Clipped')
    ax3.plot(10**loglam3,flux3,label=str(df['pmf3'][i]))
    ax3.plot(10**loglam3,weight3,':',label='weight')
    ax3.plot(10**loglam3,data3.flux,'--',label='Clipped')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()
