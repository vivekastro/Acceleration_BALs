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
from lmfit import minimize, Parameters
from astropy.modeling.models import Voigt1D

def dered_flux(Av,wave,flux):
    dered_flux = np.zeros(len(wave))
    for i in range(len(wave)):
        ext_lambda= extinction.extinction_ccm89(wave[i] * U.angstrom, a_v= Av, r_v= 3.1)
        tau_lambda= ext_lambda/(1*1.086) 
        dered_flux[i]= flux[i] * np.exp(tau_lambda)
    return dered_flux

def myVoigt(x, amp, center, fwhm_l, fwhm_g, scale, alpha):
    v1= Voigt1D(x_0=center, amplitude_L=amp, fwhm_L=fwhm_l, fwhm_G=fwhm_g)
    powerlaw = scale*x**alpha
    voigt = v1(x)
    voigt_tot = (voigt+powerlaw)
    return voigt_tot

def myGaussHermite(x, amp, center, sig, skew, kurt, scale, alpha):
    c1=-np.sqrt(3); c2=-np.sqrt(6); c3=2/np.sqrt(3); c4=np.sqrt(6)/3; c5=np.sqrt(6)/4
    gausshermite = amp*np.exp(-.5*((x-center)/sig)**2)*(1+skew*(c1*((x-center)/sig)+c3*((x-center)/sig)**3)+kurt*(c5+c2*((x-center)/sig)**2+c4*((x-center)/sig)**4))
    powerlaw = scale*x**alpha
    gaustot_gh = (gausshermite+powerlaw)
    return gaustot_gh


def myDoubleGauss(x, amp1, center1, sig1, amp2, center2, sig2, scale, alpha):
    gaus1=amp1*np.exp(-.5*((x-center1)/sig1)**2)
    gaus2=amp2*np.exp(-.5*((x-center2)/sig2)**2)
    powerlaw = scale*x**alpha
    gaustot_2g= (gaus1+gaus2+powerlaw)
    
    return gaustot_2g

def ext_coeff(lamb):
    inv_lamba=[0.45,0.61,0.8,1.82,2.27,2.7,3.22,3.34,3.46,3.6,3.75,3.92,4.09,4.28,4.50,4.73,5.00,5.24,5.38,5.52,5.70,5.88,6.07,6.27,6.48,6.72,6.98,7.23,7.52,7.84]
    smc_ext=[-2.61,-2.47,-2.12,0.0,1.0,1.67,2.29,2.65,3.0,3.15,3.49,3.91,4.24,4.53,5.3,5.85,6.38,6.76,6.9,7.17,7.71,8.01,8.49,9.06,9.28,9.84,10.8,11.51,12.52,13.54]
    xy=np.interp(1.0/(lamb*10**(-4)),inv_lamba,smc_ext)
    ext_lamb=(xy+2.98)/3.98 # Rv=2.98
    #return(z,ext_lamb)
    return(ext_lamb)

def myfunct(wave, a , b):
    return a*ext_coeff(wave)*wave**(b)


def waveRange(wave):
    #wav_range=[[1300.,1350.],[1420.,1470.],[1700.,1800.],[2080.,2215],[2480.,2655],[3225.,3900.],[4200.,4230.],[4435.,4700.],[5200.,5700.]]
    wav_range= [(1400,1850)]


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



def compute_alpha_voigt(wl, spec, ivar, wav_range, per_value=[1,99.5]):
    print ' Voigt Routine begins'
    spec[np.isnan(spec)] = 0
    ivar[np.isnan(ivar)] = 0
    print ivar
    wavelength, spectra, invar = np.array([]), np.array([]), np.array([])
    #plt.plot(wl,spec)
    for j in range(len(wav_range)):
        #print wav_range[j]
        #print min(wl),max(wl)
        temp = np.where((wl > wav_range[j][0]) & (wl < wav_range[j][1]))[0]
        #print wl[temp],len(spec),len(wl),spec[temp],ivar[temp]
        tempspec, tempivar  = spec[temp], ivar[temp]
        #print tempspec
        #print len(tempspec)
        #Mask out metal absorption lines
        cut = np.percentile(tempspec, per_value)
        #print 'cut',cut
        blah = np.where((tempspec > cut[0]) & (tempspec < cut[1])  & (tempivar > 0))[0]
        wave = wl[temp][blah]

        wavelength = np.concatenate((wavelength, wave))
        spectra = np.concatenate((spectra, tempspec[blah]))
        invar = np.concatenate((invar, tempivar[blah]))
    print 'voigt Debug',len(wavelength)
    pv0=[20.0, 1545., 50., 8.0,1.0,-1.0]
    param_boundsv = ([0,1540,0.0,0.0,0.0,-np.inf],[np.inf,1550,np.inf,20,np.inf,np.inf])

    try:
        #plt.plot(wavelength,spectra)
        #plt.show()
        poptv, pcovv = curve_fit(myVoigt, wavelength, spectra, pv0, sigma=1.0/np.sqrt(invar),bounds=param_boundsv)
    except (RuntimeError,TypeError):
        AMPv, CENTERv, SIGMALv, SIGMAGv,  SCALEv, ALPHAv, CHISQv, DOFv = np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan,  np.nan
    else:
        AMPv, CENTERv, SIGMALv, SIGMAGv,  SCALEv, ALPHAv = poptv[0], poptv[1], poptv[2], poptv[3], poptv[4], poptv[5]
        CHISQv = np.sum(invar * (spectra - myVoigt(wavelength, poptv[0], poptv[1], poptv[2], poptv[3], poptv[4], poptv[5]))**2)
        # DOF = N - n  , n = 2
        DOFv = len(spectra) - 6
    print 'Voight Routine ends' 
    print 'Compute Alpha:',  AMPv, CENTERv, SIGMALv, SIGMAGv,  SCALEv, ALPHAv, CHISQv, DOFv
    return  AMPv, CENTERv, SIGMALv, SIGMAGv,  SCALEv, ALPHAv, CHISQv, DOFv



def compute_alpha_gh(wl, spec, ivar, wav_range, per_value=[1,99.5]):
    print ' GH Routine begins'
    print ivar
    spec[np.isnan(spec)] = 0
    ivar[np.isnan(ivar)] = 0
    print ivar
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
    print '2H Debug',len(wavelength)
    pgh0=[20.0, 1545., 8., 0.2, 0.4,21.0,-1.0]
    param_boundsgh = ([0,1540,-20,-1,-np.inf,0,-np.inf],[np.inf,1555,20,1,np.inf,np.inf,np.inf])
    try:
        #plt.plot(wavelength,spectra)
        #plt.show()
        poptgh, pcovgh = curve_fit(myGaussHermite, wavelength, spectra, pgh0, sigma=1.0/np.sqrt(invar),bounds=param_boundsgh)
    except (RuntimeError, TypeError):
        AMPgh, CENTERgh, SIGMAgh, SKEWgh, KURTgh, SCALEgh, ALPHAgh, CHISQgh, DOFgh = np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan,  np.nan, np.nan
    else:
        AMPgh, CENTERgh, SIGMAgh, SKEWgh, KURTgh, SCALEgh, ALPHAgh = poptgh[0], poptgh[1], poptgh[2], poptgh[3], poptgh[4], poptgh[5], poptgh[6]
        CHISQgh = np.sum(invar * (spectra - myGaussHermite(wavelength, poptgh[0], poptgh[1], poptgh[2], poptgh[3], poptgh[4], poptgh[5], poptgh[6]))**2)
        # DOF = N - n  , n = 2
        DOFgh = len(spectra) - 7
    print 'GH Routine ends' 
    print 'Compute Alpha:', AMPgh, CENTERgh, SIGMAgh, SKEWgh, KURTgh,SCALEgh, ALPHAgh, CHISQgh, DOFgh
    return AMPgh, CENTERgh, SIGMAgh, SKEWgh, KURTgh,SCALEgh, ALPHAgh, CHISQgh, DOFgh


def compute_alpha_2g(wl, spec, ivar, wav_range, per_value=[1,99.5]):
    print '2G Routine begins'
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
    print '2G Debug',len(wavelength)
    pg20=[10.0, 1545., 9., 10.0, 1550., 18.,20.0,1]
    param_bounds2g = ([0,1540,5,0,1540,0,0,-np.inf],[np.inf,1555,np.inf,np.inf,1555,1000,np.inf,np.inf])
    try:
        #plt.plot(wavelength,spectra)
        #plt.show()
        popt2g, pcov2g = curve_fit(myDoubleGauss, wavelength, spectra, pg20, sigma=1.0/np.sqrt(invar),bounds=param_bounds2g)
    except (RuntimeError,  TypeError):
        AMPa2g, CENTERa2g, SIGMAa2g, AMPb2g, CENTERb2g, SIGMAb2g, SCALE2g, ALPHA2g, CHISQ2g, DOF2g = np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        AMPa2g, CENTERa2g, SIGMAa2g, AMPb2g, CENTERb2g, SIGMAb2g, SCALE2g, ALPHA2g = popt2g[0], popt2g[1], popt2g[2], popt2g[3], popt2g[4], popt2g[5], popt2g[6], popt2g[7]
        CHISQ2g = np.sum(invar * (spectra - myDoubleGauss(wavelength, popt2g[0], popt2g[1], popt2g[2], popt2g[3], popt2g[4], popt2g[5], popt2g[6], popt2g[7]))**2)
        # DOF = N - n  , n = 2
        DOF2g = len(spectra) - 8 
    print '2G Routine ends' 
    print 'Compute Alpha:' ,AMPa2g, CENTERa2g, SIGMAa2g, AMPb2g, CENTERb2g, SIGMAb2g, SCALE2g, ALPHA2g, CHISQ2g, DOF2g
    return AMPa2g, CENTERa2g, SIGMAa2g, AMPb2g, CENTERb2g, SIGMAb2g, SCALE2g, ALPHA2g, CHISQ2g, DOF2g 


def maskOutliers_v(wave, flux, weight, amp, center, sigmal, sigmag, scale, alpha):
    model = myVoigt(wave,amp, center, sigmal, sigmag,  scale, alpha)
    std =np.std(flux[weight > 0])
    fluxdiff = flux - model
    print 'Weights Before mask',weight
    #ww = np.where (np.abs(fluxdiff) > 3*std)
    ww = np.where (((np.abs(fluxdiff) > 2*std) & (flux <= np.median(flux))) | ((np.abs(fluxdiff) > 4*std) & (flux >= np.median(flux))))
    #nwave = np.delete(wave,ww)
    #nflux = np.delete(flux,ww)
    weight[ww] = 0#np.delete(weight,ww)
    print 'Weights After mask',weight
    return wave,flux,weight


def maskOutliers_gh(wave, flux, weight, amp, center, sigma, skew, kurt, scale, alpha):
    model = myGaussHermite(wave,amp, center, sigma, skew, kurt, scale, alpha)
    std =np.std(flux[weight > 0])
    fluxdiff = flux - model
    print 'Weights Before mask',weight
    #ww = np.where (np.abs(fluxdiff) > 3*std)
    ww = np.where (((np.abs(fluxdiff) > 2*std) & (flux <= np.median(flux))) | ((np.abs(fluxdiff) > 4*std) & (flux >= np.median(flux))))
    #nwave = np.delete(wave,ww)
    #nflux = np.delete(flux,ww)
    weight[ww] = 0#np.delete(weight,ww)
    print 'Weights After mask',weight
    return wave,flux,weight


def maskOutliers_2g(wave,flux,weight,amp1,center1,sigma1,amp2,center2,sigma2,scale,alpha):
    model = myDoubleGauss(wave,amp1,center1,sigma1,amp2,center2,sigma2,scale,alpha)
    std =np.std(flux[weight > 0])
    fluxdiff = flux - model
    #ww = np.where (np.abs(fluxdiff) > 3*std)
    ww = np.where (((np.abs(fluxdiff) > 2*std) & (flux <= np.median(flux))) | ((np.abs(fluxdiff) > 4*std) & (flux >= np.median(flux))))
    #nwave = np.delete(wave,ww)
    #nflux = np.delete(flux,ww)
    weight[ww] = 0#np.delete(weight,ww)
    
    return wave,flux,weight


def minmax(vec):
    return min(vec),max(vec)

#Version _ Tried small sigma values for both Gaussians in GausDouble and GausHermite
#Version II Tried making one of the sigmas high, one 1 and two 54 GausDouble; Same as before for GausHermite
#Version III Changing both the central wavelength to 1550 for GausDouble
df = np.genfromtxt('tdss_allmatches_crop_edit.dat',names=['ra','dec','z','pmf1','pmf2','pmf3'],dtype=(float,float,float,'|S15','|S15','|S15'))
wav_range= [(1280,1350),(1700,1800),(1950,2200),(2650,2710),(3010,3700),(3950,4050),(4140,4270)]
pp = PdfPages('ContinuumNormalization_plus_EmissionLineFits_V.pdf')
#fx=open('ALPHA_AMP_values_V.txt','w')
#fx1=open('Kate_ALPHA_AMP_values_V.txt','w')
#for i in range(len(df['pmf1'])):
for i in range(len(df)):
#for i in range(12):
    print 'Kate_Sources/spec-'+df['pmf1'][i]+'.fits' 
    print 'Kate_Sources/spec-'+df['pmf2'][i]+'.fits' 
    print 'Kate_Sources/spec-'+df['pmf3'][i]+'.fits' 
    data1 = fits.open('Kate_Sources/spec-'+df['pmf1'][i]+'.fits')[1].data
    data2 = fits.open('Kate_Sources/spec-'+df['pmf2'][i]+'.fits')[1].data
    data3 = fits.open('Kate_Sources/spec-'+df['pmf3'][i]+'.fits')[1].data
    wave1 = 10**data1.loglam.copy()
    flux1 = data1.flux.copy()
    weight1 =  (data1.ivar).copy()
    sigma1 =  1.0/np.sqrt((data1.ivar).copy())
    wave2 = 10**data2.loglam.copy()
    flux2 = data2.flux.copy()
    weight2 =  (data2.ivar).copy()
    sigma2 =  1.0/np.sqrt((data2.ivar).copy())
    wave3 = 10**data3.loglam.copy()
    flux3 = data3.flux.copy()
    weight3 =  (data3.ivar).copy()
    sigma3 =  1.0/np.sqrt((data3.ivar).copy())
    print len(wave1),len(flux1),len(weight1)
    print weight1
    print data1.and_mask
    
    #clean QSO
    #sn1= flux1*np.sqrt(weight1) ; sn2= flux2*np.sqrt(weight2); sn3= flux3*np.sqrt(weight3)
    #w1 = (weight1>0)&((sn1<-10)|(sn1>80)); w2 = (weight2>0)&((sn2<-10)|(sn2>80)) ; w3 = (weight3>0)&((sn3<-10)|(sn3>80))
    #print w1,w2,w3
    #weight1[w1] = 0; flux1[w1] = 0
    #weight2[w2] = 0; flux2[w2] = 0
    #weight3[w3] = 0; flux3[w3] = 0
    
    #de-redden the flux
    info = fits.open('Kate_Sources/spec-'+df['pmf1'][i]+'.fits')[2].data
    coords = SkyCoord(info['RA'],info['DEC'],unit='degree',frame='icrs')
    sfd = SFDQuery()
    eb_v = sfd(coords)
    dered_flux1 = dered_flux(3.1*eb_v,wave1,flux1)
    dered_flux2 = dered_flux(3.1*eb_v,wave2,flux2)
    dered_flux3 = dered_flux(3.1*eb_v,wave3,flux3)
    
    # Change wavelengths to rest wavellegths
    rwave1 = wave1/(1.0+df['z'][i])
    rwave2 = wave2/(1.0+df['z'][i])
    rwave3 = wave3/(1.0+df['z'][i])
    
    #Clip all the spectra to same wavelength bounds Necessary to capture variation in alpha
    wmin1,wmax1 = minmax(rwave1) 
    wmin2,wmax2 = minmax(rwave2) 
    wmin3,wmax3 = minmax(rwave3)
    wlim1,wlim2 = max([wmin1,wmin2,wmin3]),min([wmax1,wmax2,wmax3])
    l1 = np.where((rwave1 >= 1410) & (rwave1 <= 1650))[0]
    l2 = np.where((rwave2 >= 1410) & (rwave2 <= 1650))[0]
    l3 = np.where((rwave3 >= 1410) & (rwave3 <= 1650))[0]
    crwave1 = rwave1[l1];cdered_flux1 = dered_flux1[l1];cweight1 = weight1[l1];csigma1= sigma1[l1]
    crwave2 = rwave2[l2];cdered_flux2 = dered_flux2[l2];cweight2 = weight2[l2];csigma2 = sigma2[l2]
    crwave3 = rwave3[l3];cdered_flux3 = dered_flux3[l3];cweight3 = weight3[l3];csigma3 = sigma3[l3]

    #Initial Mask for broad absorptions
    cut1 = np.nanpercentile(cdered_flux1, 20)
    cut2 = np.nanpercentile(cdered_flux2, 20)
    cut3 = np.nanpercentile(cdered_flux3, 20)
    xx1 = np.where(cdered_flux1 < cut1)[0]
    xx2 = np.where(cdered_flux2 < cut2)[0]
    xx3 = np.where(cdered_flux3 < cut3)[0]
    print cut1
    print dered_flux1[xx1]
    cweight1[xx1] = 0
    cweight2[xx2] = 0
    cweight3[xx3] = 0
    
        
    #Fit powerlaw iterate and mask outliers
    iteration = 3
    for j in range(iteration):
        if j == 0:
            print 'iteration 1 Begins'
            print j,'GH',df['pmf1'][i]
            AMPgh1, CENTERgh1, SIGMAgh1, SKEWgh1, KURTgh1, SCALEgh1, ALPHAgh1, CHISQgh1, DOFgh1   = compute_alpha_gh(crwave1, cdered_flux1, cweight1,  waveRange(crwave1))
            print j,'2G',df['pmf1'][i]
            AMPa2g1, CENTERa2g1, SIGMAa2g1, AMPb2g1, CENTERb2g1, SIGMAb2g1, SCALE2g1, ALPHA2g1 ,CHISQ2g1, DOF2g1   = compute_alpha_2g(crwave1, cdered_flux1, cweight1,  waveRange(crwave1))
            print j,'V',df['pmf1'][i]
            AMPv1, CENTERv1, SIGMALv1, SIGMAGv1,  SCALEv1, ALPHAv1, CHISQv1, DOFv1   = compute_alpha_voigt(crwave1, cdered_flux1, cweight1,  waveRange(crwave1))
            print 'Object 1 Done'
            print j,'GH',df['pmf2'][i]
            AMPgh2, CENTERgh2, SIGMAgh2, SKEWgh2, KURTgh2, SCALEgh2, ALPHAgh2, CHISQgh2, DOFgh2   = compute_alpha_gh(crwave2, cdered_flux2, cweight2,  waveRange(crwave2))
            print j,'2G',df['pmf2'][i]
            AMPa2g2, CENTERa2g2, SIGMAa2g2, AMPb2g2, CENTERb2g2, SIGMAb2g2, SCALE2g2, ALPHA2g2 ,CHISQ2g2, DOF2g2   = compute_alpha_2g(crwave2, cdered_flux2, cweight2,  waveRange(crwave2))
            print j,'V',df['pmf2'][i]
            AMPv2, CENTERv2, SIGMALv2, SIGMAGv2,  SCALEv2, ALPHAv2, CHISQv2, DOFv2   = compute_alpha_voigt(crwave2, cdered_flux2, cweight2,  waveRange(crwave2))
            print 'Object 2 Done'
            print j,'GH',df['pmf3'][i]
            AMPgh3, CENTERgh3, SIGMAgh3, SKEWgh3, KURTgh3, SCALEgh3, ALPHAgh3, CHISQgh3, DOFgh3   = compute_alpha_gh(crwave3, cdered_flux3, cweight3,  waveRange(crwave3))
            print j,'2G',df['pmf3'][i]
            AMPa2g3, CENTERa2g3, SIGMAa2g3, AMPb2g3, CENTERb2g3, SIGMAb2g3, SCALE2g3, ALPHA2g3 ,CHISQ2g3, DOF2g3   = compute_alpha_2g(crwave3, cdered_flux3, cweight3,  waveRange(crwave3))
            print j,'V',df['pmf3'][i]
            AMPv3, CENTERv3, SIGMALv3, SIGMAGv3,  SCALEv3, ALPHAv3, CHISQv3, DOFv3   = compute_alpha_voigt(crwave3, cdered_flux3, cweight3,  waveRange(crwave3))
            print 'Object 3 Done'
            nwavegh1 = crwave1; nfluxgh1=cdered_flux1;nweightgh1 = cweight1
            nwavegh2 = crwave2; nfluxgh2=cdered_flux2;nweightgh2 = cweight2
            nwavegh3 = crwave3; nfluxgh3=cdered_flux3;nweightgh3 = cweight3

            nwave2g1 = crwave1; nflux2g1=cdered_flux1;nweight2g1 = cweight1
            nwave2g2 = crwave2; nflux2g2=cdered_flux2;nweight2g2 = cweight2
            nwave2g3 = crwave3; nflux2g3=cdered_flux3;nweight2g3 = cweight3


            nwavev1 = crwave1; nfluxv1=cdered_flux1;nweightv1 = cweight1
            nwavev2 = crwave2; nfluxv2=cdered_flux2;nweightv2 = cweight2
            nwavev3 = crwave3; nfluxv3=cdered_flux3;nweightv3 = cweight3
            print 'iteration 1 Ends'
            continue
        else:
            nwavegh1,nfluxgh1,nweightgh1 = maskOutliers_gh(nwavegh1, nfluxgh1, nweightgh1, AMPgh1, CENTERgh1, SIGMAgh1, SKEWgh1, KURTgh1, SCALEgh1, ALPHAgh1)
            nwavegh2,nfluxgh2,nweightgh2 = maskOutliers_gh(nwavegh2, nfluxgh2, nweightgh2, AMPgh2, CENTERgh2, SIGMAgh2, SKEWgh2, KURTgh2, SCALEgh2, ALPHAgh2)
            nwavegh3,nfluxgh3,nweightgh3 = maskOutliers_gh(nwavegh3, nfluxgh3, nweightgh3, AMPgh3, CENTERgh3, SIGMAgh3, SKEWgh3, KURTgh3, SCALEgh3, ALPHAgh3)
            
            nwave2g1,nflux2g1,nweight2g1 = maskOutliers_2g(nwave2g1, nflux2g1, nweight2g1, AMPa2g1, CENTERa2g1, SIGMAa2g1, AMPb2g1, CENTERb2g1, SIGMAb2g1, SCALE2g1, ALPHA2g1)
            nwave2g2,nflux2g2,nweight2g2 = maskOutliers_2g(nwave2g2, nflux2g2, nweight2g2, AMPa2g2, CENTERa2g2, SIGMAa2g2, AMPb2g2, CENTERb2g2, SIGMAb2g2, SCALE2g2, ALPHA2g2)
            nwave2g3,nflux2g3,nweight2g3 = maskOutliers_2g(nwave2g3, nflux2g3, nweight2g3, AMPa2g3, CENTERa2g3, SIGMAa2g3, AMPb2g3, CENTERb2g3, SIGMAb2g3, SCALE2g3, ALPHA2g3)

            nwavev1,nfluxv1,nweightv1 = maskOutliers_v(nwavev1, nfluxv1, nweightv1, AMPv1, CENTERv1, SIGMALv1, SIGMAGv1,  SCALEv1, ALPHAv1 )
            nwavev2,nfluxv2,nweightv2 = maskOutliers_v(nwavev2, nfluxv2, nweightv2, AMPv2, CENTERv2, SIGMALv2, SIGMAGv2,  SCALEv2, ALPHAv2 )
            nwavev3,nfluxv3,nweightv3 = maskOutliers_v(nwavev3, nfluxv3, nweightv3, AMPv3, CENTERv3, SIGMALv3, SIGMAGv3,  SCALEv3, ALPHAv3 )

            
            print j,'GH',df['pmf1'][i]
            AMPgh1, CENTERgh1, SIGMAgh1, SKEWgh1, KURTgh1, SCALEgh1, ALPHAgh1, CHISQgh1, DOFgh1   = compute_alpha_gh(nwavegh1, nfluxgh1, nweightgh1,  waveRange(nwavegh1))
            print j,'2G',df['pmf1'][i]
            AMPa2g1, CENTERa2g1, SIGMAa2g1, AMPb2g1, CENTERb2g1, SIGMAb2g1, SCALE2g1, ALPHA2g1 ,CHISQ2g1, DOF2g1   = compute_alpha_2g(nwave2g1, nflux2g1, nweight2g1,  waveRange(nwave2g1))
            print j,'V',df['pmf1'][i]
            AMPv1, CENTERv1, SIGMALv1, SIGMAGv1,  SCALEv1, ALPHAv1, CHISQv1, DOFv1   = compute_alpha_voigt(nwavev1, nfluxv1, nweightv1,  waveRange(nwavev1))
            print j,'GH',df['pmf2'][i]
            AMPgh2, CENTERgh2, SIGMAgh2, SKEWgh2, KURTgh2, SCALEgh2, ALPHAgh2, CHISQgh2, DOFgh2   = compute_alpha_gh(nwavegh2, nfluxgh2, nweightgh2,  waveRange(nwavegh2))
            print j,'2G',df['pmf2'][i]
            AMPa2g2, CENTERa2g2, SIGMAa2g2, AMPb2g2, CENTERb2g2, SIGMAb2g2, SCALE2g2, ALPHA2g2 ,CHISQ2g2, DOF2g2   = compute_alpha_2g(nwave2g2, nflux2g2, nweight2g2,  waveRange(nwave2g2))
            print j,'V',df['pmf2'][i]
            AMPv2, CENTERv2, SIGMALv2, SIGMAGv2,  SCALEv2, ALPHAv2, CHISQv2, DOFv2   = compute_alpha_voigt(nwavev2, nfluxv2, nweightv2,  waveRange(nwavev2))
            print j,'GH',df['pmf3'][i]
            AMPgh3, CENTERgh3, SIGMAgh3, SKEWgh3, KURTgh3, SCALEgh3, ALPHAgh3, CHISQgh3, DOFgh3   = compute_alpha_gh(nwavegh3, nfluxgh3, nweightgh3,  waveRange(nwavegh3))
            print j,'2G',df['pmf3'][i]
            AMPa2g3, CENTERa2g3, SIGMAa2g3, AMPb2g3, CENTERb2g3, SIGMAb2g3, SCALE2g3, ALPHA2g3 ,CHISQ2g3, DOF2g3   = compute_alpha_2g(nwave2g3, nflux2g3, nweight2g3,  waveRange(nwave2g3))
            print j,'V',df['pmf3'][i]
            AMPv3, CENTERv3, SIGMALv3, SIGMAGv3,  SCALEv3, ALPHAv3, CHISQv3, DOFv3   = compute_alpha_voigt(nwavev3, nfluxv3, nweightv3,  waveRange(nwavev3))



    fig,((ax1,rax1),(ax2,rax2),(ax3,rax3))=plt.subplots(3,2,figsize=(20,10))
    ax1.plot(crwave1,cdered_flux1)
    ax1.plot(crwave1,cweight1,alpha=0.2)
    ax1.plot(crwave1[cweight1 > 0],cdered_flux1[cweight1>0],'.')
    ax2.plot(crwave2,cdered_flux2)
    ax2.plot(crwave2,cweight2,alpha=0.2)
    ax2.plot(crwave2[cweight2 > 0],cdered_flux2[cweight2>0],'.')

    ax3.plot(crwave3,cdered_flux3)
    ax3.plot(crwave3,cweight3,alpha=0.2)
    ax3.plot(crwave3[cweight3 > 0],cdered_flux3[cweight3>0],'.')

    #ax1.plot(cwave1,cflux1,'--')
    pgh1=ax1.plot(crwave1,myGaussHermite(crwave1,AMPgh1, CENTERgh1, SIGMAgh1, SKEWgh1, KURTgh1, SCALEgh1, ALPHAgh1),':',color='red',lw=3,label='Gauss-Hermite')
    pgh2=ax2.plot(crwave2,myGaussHermite(crwave2,AMPgh2, CENTERgh2, SIGMAgh2, SKEWgh2, KURTgh2, SCALEgh2, ALPHAgh2),':',color='red',lw=3,label='Gauss-Hermite')
    pgh3=ax3.plot(crwave3,myGaussHermite(crwave3,AMPgh3, CENTERgh3, SIGMAgh3, SKEWgh3, KURTgh3, SCALEgh3, ALPHAgh3),':',color='red',lw=3,label='Gauss-Hermite')
    
    p2g1=ax1.plot(crwave1,myDoubleGauss(crwave1,AMPa2g1, CENTERa2g1, SIGMAa2g1, AMPb2g1, CENTERb2g1, SIGMAb2g1, SCALE2g1, ALPHA2g1 ),':',color='blue',lw=3,label='2-Gaussian')
    p2g2=ax2.plot(crwave2,myDoubleGauss(crwave2,AMPa2g2, CENTERa2g2, SIGMAa2g2, AMPb2g2, CENTERb2g2, SIGMAb2g2, SCALE2g2, ALPHA2g2 ),':',color='blue',lw=3,label='2-Gaussian')
    p2g3=ax3.plot(crwave1,myDoubleGauss(crwave3,AMPa2g3, CENTERa2g3, SIGMAa2g3, AMPb2g3, CENTERb2g3, SIGMAb2g3, SCALE2g3, ALPHA2g3 ),':',color='blue',lw=3,label='2-Gaussian')
    
    p2v1 = ax1.plot(crwave1,myVoigt(crwave1,AMPv1, CENTERv1, SIGMALv1, SIGMAGv1,  SCALEv1, ALPHAv1),':',color='orange',lw=3,label='Voigt')
    p2v2 = ax2.plot(crwave2,myVoigt(crwave2,AMPv2, CENTERv2, SIGMALv2, SIGMAGv2,  SCALEv2, ALPHAv2),':',color='orange',lw=3,label='Voigt')
    p2v3 = ax3.plot(crwave3,myVoigt(crwave3,AMPv3, CENTERv3, SIGMALv3, SIGMAGv3,  SCALEv3, ALPHAv3),':',color='orange',lw=3,label='Voigt')
    ax1.axhline(cut1,ls=':')
    ax2.axhline(cut1,ls=':')
    ax3.axhline(cut1,ls=':')
    string1gh = 'AMP: {0:4.3f}   CENTER: {1:4.3f} RCHIS:{2:4.3f}'.format(AMPgh1,CENTERgh1,(CHISQgh1/DOFgh1))
    string12g = 'AMP: {0:4.3f}   CENTER: {1:4.3f} RCHIS:{2:4.3f}'.format(AMPa2g1,CENTERa2g1,(CHISQ2g1/ DOF2g1))
    string2gh = 'AMP: {0:4.3f}   CENTER: {1:4.3f} RCHIS:{2:4.3f}'.format(AMPgh2,CENTERgh2,(CHISQgh2/ DOFgh2))
    string22g = 'AMP: {0:4.3f}   CENTER: {1:4.3f} RCHIS:{2:4.3f}'.format(AMPa2g2,CENTERa2g2,(CHISQ2g2/ DOF2g2))
    string3gh = 'AMP: {0:4.3f}   CENTER: {1:4.3f} RCHIS:{2:4.3f}'.format(AMPgh3,CENTERgh3,(CHISQgh3/ DOFgh3))
    string32g = 'AMP: {0:4.3f}   CENTER: {1:4.3f} RCHIS:{2:4.3f}'.format(AMPa2g3,CENTERa2g3,(CHISQ2g3/ DOF2g3))
    ax1.set_xlim(np.min(crwave1),np.max(crwave1))
    ax2.set_xlim(np.min(crwave1),np.max(crwave1))
    ax3.set_xlim(np.min(crwave1),np.max(crwave1))
    xlim=ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ylim2 = ax1.get_ylim()
    ylim3 = ax1.get_ylim()
    ax1.set_ylim(min(savitzky_golay(cdered_flux1,101,2))-3*np.std(cdered_flux1),max(savitzky_golay(cdered_flux1,101,2))+5*np.std(cdered_flux1))
    ax2.set_ylim(min(savitzky_golay(cdered_flux2,101,2))-3*np.std(cdered_flux2),max(savitzky_golay(cdered_flux2,101,2))+5*np.std(cdered_flux2))
    ax3.set_ylim(min(savitzky_golay(cdered_flux3,101,2))-3*np.std(cdered_flux3),max(savitzky_golay(cdered_flux3,101,2))+5*np.std(cdered_flux3))
    ax1.annotate(str(df['pmf1'][i]) ,xy=(.95,.15), xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    ax2.annotate(str(df['pmf2'][i]),xy=(.95,.15), xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    ax3.annotate(str(df['pmf3'][i]),xy=(.95,.15), xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    #ax1.text(xlim[0]+0.1*(xlim[1] - xlim[0]),ylim1[1] - 0.2*(ylim1[1] - ylim1[0]),string12g)
    #ax2.text(xlim[0]+0.1*(xlim[1] - xlim[0]),ylim2[1] - 0.1*(ylim1[1] - ylim2[0]),string2gh)
    #ax2.text(xlim[0]+0.1*(xlim[1] - xlim[0]),ylim2[1] - 0.2*(ylim1[1] - ylim2[0]),string22g)
    #ax3.text(xlim[0]+0.1*(xlim[1] - xlim[0]),ylim3[1] - 0.1*(ylim1[1] - ylim3[0]),string3gh)
    #ax3.text(xlim[0]+0.1*(xlim[1] - xlim[0]),ylim3[1] - 0.2*(ylim1[1] - ylim2[0]),string32g)
    #ax1.legend([pgh1,p2g1,p2v1],['Gaus-Hermite','2-Gaus','Voigt'],loc=3)
    ax1.legend(loc=3)
    ax1.annotate('Gauss-Hermite:\nAmp = %.2f\nCenter = %.2f\n$\sigma$ = %.2f\nH3 = %.2f\nH4 = %.2f\nrChi2 = %.3f' \
        %(AMPgh1, CENTERgh1, SIGMAgh1, SKEWgh1, KURTgh1,(CHISQgh1/ DOFgh1)),xy=(.05,.95), \
        xycoords='axes fraction',ha="left", va="top", \
    bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    ax1.annotate('Double Gaussian:\nAmp$_1$ = %.2f\nAmp$_2$ = %.2f\nCenter$_1$ = %.2f\nCenter$_2$ = %.2f\n$\sigma_1$ = %.2f\n$\sigma_2$ = %.2f\nrChi2 = %.3f' \
        %(AMPa2g1, AMPb2g1, CENTERa2g1, CENTERb2g1, SIGMAa2g1, SIGMAb2g1,(CHISQ2g1/ DOF2g1)),xy=(.95,.95), \
        xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    ax1.annotate('Voigt:\nAmp$_1$ = %.2f\nCenter = %.2f\n$\sigma_l$ = %.2f\n$\sigma_g$ = %.2f\nrChi2 = %.3f' \
        %(AMPv1, CENTERv1, SIGMALv1, SIGMAGv1,(CHISQv1/ DOFv1)),xy=(.35,.95), \
        xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)

    ax2.legend(loc=3)
    #ax2.legend([pgh2,p2g2,p2v2],['Gaus-Hermite','2-Gaus','Voigt'],prop={'size':8},loc='center left')
    ax2.annotate('Gauss-Hermite:\nAmp = %.2f\nCenter = %.2f\n$\sigma$ = %.2f\nH3 = %.2f\nH4 = %.2f\nrChi2 = %.3f'\
        %(AMPgh2, CENTERgh2, SIGMAgh2, SKEWgh2, KURTgh2,(CHISQgh2/ DOFgh2)),xy=(.05,.95), \
        xycoords='axes fraction',ha="left", va="top", \
    bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    ax2.annotate('Double Gaussian:\nAmp$_1$ = %.2f\nAmp$_2$ = %.2f\nCenter$_1$ = %.2f\nCenter$_2$ = %.2f\n$\sigma_1$ = %.2f\n$\sigma_2$ = %.2f\nrChi2 = %.3f' \
        %(AMPa2g2, AMPb2g2, CENTERa2g2, CENTERb2g2, SIGMAa2g2, SIGMAb2g2,(CHISQ2g2/ DOF2g2)),xy=(.95,.95), \
        xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    ax2.annotate('Voigt:\nAmp$_1$ = %.2f\nCenter = %.2f\n$\sigma_l$ = %.2f\n$\sigma_g$ = %.2f\nrChi2 = %.3f' \
        %(AMPv2, CENTERv2, SIGMALv2, SIGMAGv2,(CHISQv2/ DOFv2)),xy=(.35,.95), \
        xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)

    ax3.legend(loc=3)
    #ax3.legend([pgh3,p2g3,p2v3],['Gaus-Hermite','2-Gaus','Voigt'],prop={'size':8},loc='center left')
    ax3.annotate('Gauss-Hermite:\nAmp = %.2f\nCenter = %.2f\n$\sigma$ = %.2f\nH3 = %.2f\nH4 = %.2f\nrChi2 = %.3f' \
        %(AMPgh3, CENTERgh3, SIGMAgh3, SKEWgh3, KURTgh3,(CHISQgh3/ DOFgh3)),xy=(.05,.95), \
        xycoords='axes fraction',ha="left", va="top", \
    bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    ax3.annotate('Double Gaussian:\nAmp$_1$ = %.2f\nAmp$_2$ = %.2f\nCenter$_1$ = %.2f\nCenter$_2$ = %.2f\n$\sigma_1$ = %.2f\n$\sigma_2$ = %.2f\nrChi2 = %.3f' \
        %(AMPa2g3, AMPb2g3, CENTERa2g3, CENTERb2g3, SIGMAa2g3, SIGMAb2g3,(CHISQ2g3/ DOF2g3)),xy=(.95,.95), \
        xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    ax3.annotate('Voigt:\nAmp$_1$ = %.2f\nCenter = %.2f\n$\sigma_l$ = %.2f\n$\sigma_g$ = %.2f\nrChi2 = %.3f' \
        %(AMPv3, CENTERv3, SIGMALv3, SIGMAGv3,(CHISQv3/ DOFv3)),xy=(.35,.95), \
        xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    
    ax1.set_ylabel('Flux')
    ax1.set_xlabel('Wavelength ($\AA$)')
    ax2.set_ylabel('Flux')
    ax2.set_xlabel('Wavelength ($\AA$)')
    ax3.set_ylabel('Flux')
    ax3.set_xlabel('Wavelength ($\AA$)')
    #Save the spectra
    filenamegh1 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf1'][i]+'_cEm_gh.txt'
    filenamegh2 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf2'][i]+'_cEm_gh.txt'
    filenamegh3 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf3'][i]+'_cEm_gh.txt'
    
    filename2g1 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf1'][i]+'_cEm_2g.txt'
    filename2g2 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf2'][i]+'_cEm_2g.txt'
    filename2g3 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf3'][i]+'_cEm_2g.txt'

    filenamev1 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf1'][i]+'_cEm_v.txt'
    filenamev2 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf2'][i]+'_cEm_v.txt'
    filenamev3 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf3'][i]+'_cEm_v.txt'


    # Residuals & Weights
    resgh1 = cdered_flux1 / myGaussHermite(crwave1,AMPgh1, CENTERgh1, SIGMAgh1, SKEWgh1, KURTgh1, SCALEgh1, ALPHAgh1)
    resgh2 = cdered_flux2 / myGaussHermite(crwave2,AMPgh2, CENTERgh2, SIGMAgh2, SKEWgh2, KURTgh2, SCALEgh2, ALPHAgh2)
    resgh3 = cdered_flux3 / myGaussHermite(crwave3,AMPgh3, CENTERgh3, SIGMAgh3, SKEWgh3, KURTgh3, SCALEgh3, ALPHAgh3)
    
    wresgh1 = csigma1 / myGaussHermite(crwave1,AMPgh1, CENTERgh1, SIGMAgh1, SKEWgh1, KURTgh1, SCALEgh1, ALPHAgh1)
    wresgh2 = csigma2 / myGaussHermite(crwave2,AMPgh2, CENTERgh2, SIGMAgh2, SKEWgh2, KURTgh2, SCALEgh2, ALPHAgh2)
    wresgh3 = csigma3 / myGaussHermite(crwave3,AMPgh3, CENTERgh3, SIGMAgh3, SKEWgh3, KURTgh3, SCALEgh3, ALPHAgh3)

    res2g1 = cdered_flux1 / myDoubleGauss(crwave1,AMPa2g1, CENTERa2g1, SIGMAa2g1, AMPb2g1, CENTERb2g1, SIGMAb2g1, SCALE2g1, ALPHA2g1 )
    res2g2 = cdered_flux2 / myDoubleGauss(crwave2,AMPa2g2, CENTERa2g2, SIGMAa2g2, AMPb2g2, CENTERb2g2, SIGMAb2g2, SCALE2g2, ALPHA2g2 )
    res2g3 = cdered_flux3 / myDoubleGauss(crwave3,AMPa2g3, CENTERa2g3, SIGMAa2g3, AMPb2g3, CENTERb2g3, SIGMAb2g3, SCALE2g3, ALPHA2g3 )

    wres2g1 = csigma1 / myDoubleGauss(crwave1,AMPa2g1, CENTERa2g1, SIGMAa2g1, AMPb2g1, CENTERb2g1, SIGMAb2g1, SCALE2g1, ALPHA2g1 )
    wres2g2 = csigma2 / myDoubleGauss(crwave2,AMPa2g2, CENTERa2g2, SIGMAa2g2, AMPb2g2, CENTERb2g2, SIGMAb2g2, SCALE2g2, ALPHA2g2 )
    wres2g3 = csigma3 / myDoubleGauss(crwave3,AMPa2g3, CENTERa2g3, SIGMAa2g3, AMPb2g3, CENTERb2g3, SIGMAb2g3, SCALE2g3, ALPHA2g3 )

    resv1 = cdered_flux1 / myVoigt(crwave1,AMPv1, CENTERv1, SIGMALv1, SIGMAGv1,  SCALEv1, ALPHAv1)
    resv2 = cdered_flux2 / myVoigt(crwave2,AMPv2, CENTERv2, SIGMALv2, SIGMAGv2,  SCALEv2, ALPHAv2)
    resv3 = cdered_flux3 / myVoigt(crwave3,AMPv3, CENTERv3, SIGMALv3, SIGMAGv3,  SCALEv3, ALPHAv3)

    wresv1 = csigma1 / myVoigt(crwave1,AMPv1, CENTERv1, SIGMALv1, SIGMAGv1,  SCALEv1, ALPHAv1)
    wresv2 = csigma2 / myVoigt(crwave2,AMPv2, CENTERv2, SIGMALv2, SIGMAGv2,  SCALEv2, ALPHAv2)
    wresv3 = csigma3 / myVoigt(crwave3,AMPv3, CENTERv3, SIGMALv3, SIGMAGv3,  SCALEv3, ALPHAv3)

    #Save Begins
    np.savetxt(filenamegh1,zip(crwave1,resgh1,wresgh1), fmt='%10.5f')
    np.savetxt(filenamegh2,zip(crwave2,resgh2,wresgh2), fmt='%10.5f')
    np.savetxt(filenamegh3,zip(crwave3,resgh3,wresgh3), fmt='%10.5f')

    np.savetxt(filename2g1,zip(crwave1,res2g1,wres2g1), fmt='%10.5f')
    np.savetxt(filename2g2,zip(crwave2,res2g2,wres2g2), fmt='%10.5f')
    np.savetxt(filename2g3,zip(crwave3,res2g3,wres2g3), fmt='%10.5f')

    np.savetxt(filenamev1,zip(crwave1,resv1,wresv1), fmt='%10.5f')
    np.savetxt(filenamev2,zip(crwave2,resv2,wresv2), fmt='%10.5f')
    np.savetxt(filenamev3,zip(crwave3,resv3,wresv3), fmt='%10.5f')

    #Plot Begins
    presgh1,pres2g1,presv1,=rax1.plot(crwave1,resgh1,'k--',crwave1,res2g1,'r-',crwave1,resv1,'b:',alpha=0.7)
    presgh2,pres2g2,presv2,=rax2.plot(crwave2,resgh2,'k--',crwave2,res2g2,'r-',crwave2,resv2,'b:',alpha=0.7)
    presgh3,pres2g3,presv3,=rax3.plot(crwave3,resgh3,'k--',crwave3,res2g3,'r-',crwave3,resv3,'b:',alpha=0.7)

    rax1.set_ylabel('Normalized Flux')
    rax1.set_xlabel('Wavelength ($\AA$)')
    rax1.legend([presgh1,pres2g1,presv1],['Gaus-Hermite','2-Gaus','Voigt'],numpoints=4,prop={'size':8},loc='lower right')
    rax2.set_ylabel('Normalized Flux')
    rax2.set_xlabel('Wavelength ($\AA$)')
    rax2.legend([presgh2,pres2g2,presv2],['Gaus-Hermite','2-Gaus','Voigt'],numpoints=4,prop={'size':8},loc='lower right')
    rax3.set_ylabel('Normalized Flux')
    rax3.set_xlabel('Wavelength ($\AA$)')
    rax3.legend([presgh3,pres2g3,presv3],['Gaus-Hermite','2-Gaus', 'Voigt'],numpoints=4,prop={'size':8},loc='lower right')
    rax1.axhline(1.0,ls=':',color='blue',alpha=0.5)
    rax2.axhline(1.0,ls=':',color='blue',alpha=0.5)
    rax3.axhline(1.0,ls=':',color='blue',alpha=0.5)
    rax1.set_ylim(0,2)
    rax2.set_ylim(0,2)
    rax3.set_ylim(0,2)
    fig.tight_layout()
    fig.savefig(pp,format='pdf')
    #plt.show()
    plt.clf()
pp.close()

