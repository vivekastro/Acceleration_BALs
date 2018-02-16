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
    pv0=[1.0, 1545., 50., 8.0,1.0,1.0]
    param_boundsv = ([0,1540,0.0,0.0,0.0,-np.inf],[np.inf,1550,np.inf,20,np.inf,np.inf])

    try:
        #plt.plot(wavelength,spectra)
        #plt.show()
        poptv, pcovv = curve_fit(myVoigt, wavelength, spectra, pv0, sigma=1.0/np.sqrt(invar),bounds=param_boundsv)
    except (RuntimeError, TypeError):
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
    pgh0=[1.0, 1545., 8., 0.2, 0.4,1.0,1.0]
    param_boundsgh = ([0,1540,-20,-np.inf,-np.inf,0,-np.inf],[np.inf,1555,20,np.inf,np.inf,np.inf,np.inf])
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
    pg20=[1.0, 1545., 8., 1.0, 1550., 54.,1.0,1.0]
    param_bounds2g = ([0,1540,0,0,1540,-np.inf,-np.inf,-np.inf],[np.inf,1555,np.inf,np.inf,1555,np.inf,np.inf,np.inf])
    try:
        #plt.plot(wavelength,spectra)
        #plt.show()
        popt2g, pcov2g = curve_fit(myDoubleGauss, wavelength, spectra, pg20, sigma=1.0/np.sqrt(invar),bounds=param_bounds2g)
    except (RuntimeError, TypeError):
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
pp = PdfPages('ContinuumNormalization_plus_EmissionLineFits_IV.pdf')
#fx=open('ALPHA_AMP_values_V.txt','w')
#fx1=open('Kate_ALPHA_AMP_values_V.txt','w')
#for i in range(len(df['pmf1'])):
for i in range(len(df)):
#for i in range(12):

    print 'Norm_Spectra/Normspec_'+df['pmf1'][i]+'.txt' 
    print 'Norm_Spectra/Normspec_'+df['pmf2'][i]+'.txt' 
    print 'Norm_Spectra/Normspec_'+df['pmf3'][i]+'.txt' 
    if ((not os.path.isfile('Norm_Spectra/Normspec_'+df['pmf1'][i]+'.txt')) | (not os.path.isfile('Norm_Spectra/Normspec_'+df['pmf2'][i]+'.txt')) | ( not os.path.isfile('Norm_Spectra/Normspec_'+df['pmf3'][i]+'.txt'))):
        continue
    else:
        data1 = np.loadtxt('Norm_Spectra/Normspec_'+df['pmf1'][i]+'.txt')
        data2 = np.loadtxt('Norm_Spectra/Normspec_'+df['pmf2'][i]+'.txt')
        data3 = np.loadtxt('Norm_Spectra/Normspec_'+df['pmf3'][i]+'.txt')
    wave1 = data1.T[0] ;    flux1 = data1.T[1] ; weight1 =  data1.T[2]; mask1 = data1.T[3]
    wave2 = data2.T[0] ;    flux2 = data2.T[1] ; weight2 =  data2.T[2]; mask2 = data2.T[3]
    wave3 = data3.T[0] ;    flux3 = data3.T[1] ; weight3 =  data3.T[2]; mask3 = data3.T[3]
    print len(wave1),len(flux1),len(weight1)
    #print weight1
    #Initial Mask for broad absorptions
    cut1 = np.nanpercentile(flux1, 20)
    cut2 = np.nanpercentile(flux2, 20)
    cut3 = np.nanpercentile(flux3, 20)
    xx1 = np.where(flux1 < cut1)[0]
    xx2 = np.where(flux2 < cut2)[0]
    xx3 = np.where(flux3 < cut3)[0]
    print cut1
    print flux1[xx1]
    weight1[xx1] = 0
    weight2[xx2] = 0
    weight3[xx3] = 0
    
        
    #Fit powerlaw iterate and mask outliers
    iteration = 3
    for j in range(iteration):
        if j == 0:
            print 'iteration 1 Begins'
            AMPgh1, CENTERgh1, SIGMAgh1, SKEWgh1, KURTgh1, SCALEgh1, ALPHAgh1, CHISQgh1, DOFgh1   = compute_alpha_gh(wave1, flux1, weight1,  waveRange(wave1))
            AMPa2g1, CENTERa2g1, SIGMAa2g1, AMPb2g1, CENTERb2g1, SIGMAb2g1, SCALE2g1, ALPHA2g1 ,CHISQ2g1, DOF2g1   = compute_alpha_2g(wave1, flux1, weight1,  waveRange(wave1))
            AMPv1, CENTERv1, SIGMALv1, SIGMAGv1,  SCALEv1, ALPHAv1, CHISQv1, DOFv1   = compute_alpha_voigt(wave1, flux1, weight1,  waveRange(wave1))
            print 'Object 1 Done'
            AMPgh2, CENTERgh2, SIGMAgh2, SKEWgh2, KURTgh2, SCALEgh2, ALPHAgh2, CHISQgh2, DOFgh2   = compute_alpha_gh(wave2, flux2, weight2,  waveRange(wave2))
            AMPa2g2, CENTERa2g2, SIGMAa2g2, AMPb2g2, CENTERb2g2, SIGMAb2g2, SCALE2g2, ALPHA2g2 ,CHISQ2g2, DOF2g2   = compute_alpha_2g(wave2, flux2, weight2,  waveRange(wave2))
            AMPv2, CENTERv2, SIGMALv2, SIGMAGv2,  SCALEv2, ALPHAv2, CHISQv2, DOFv2   = compute_alpha_voigt(wave2, flux2, weight2,  waveRange(wave2))
            print 'Object 2 Done'
            AMPgh3, CENTERgh3, SIGMAgh3, SKEWgh3, KURTgh3, SCALEgh3, ALPHAgh3, CHISQgh3, DOFgh3   = compute_alpha_gh(wave3, flux3, weight3,  waveRange(wave3))
            AMPa2g3, CENTERa2g3, SIGMAa2g3, AMPb2g3, CENTERb2g3, SIGMAb2g3, SCALE2g3, ALPHA2g3 ,CHISQ2g3, DOF2g3   = compute_alpha_2g(wave3, flux3, weight3,  waveRange(wave3))
            AMPv3, CENTERv3, SIGMALv3, SIGMAGv3,  SCALEv3, ALPHAv3, CHISQv3, DOFv3   = compute_alpha_voigt(wave3, flux3, weight3,  waveRange(wave3))
            nwavegh1 = wave1; nfluxgh1=flux1;nweightgh1 = weight1
            nwavegh2 = wave2; nfluxgh2=flux2;nweightgh2 = weight2
            nwavegh3 = wave3; nfluxgh3=flux3;nweightgh3 = weight3

            nwave2g1 = wave1; nflux2g1=flux1;nweight2g1 = weight1
            nwave2g2 = wave2; nflux2g2=flux2;nweight2g2 = weight2
            nwave2g3 = wave3; nflux2g3=flux3;nweight2g3 = weight3


            nwavev1 = wave1; nfluxv1=flux1;nweightv1 = weight1
            nwavev2 = wave2; nfluxv2=flux2;nweightv2 = weight2
            nwavev3 = wave3; nfluxv3=flux3;nweightv3 = weight3
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

            
            AMPgh1, CENTERgh1, SIGMAgh1, SKEWgh1, KURTgh1, SCALEgh1, ALPHAgh1, CHISQgh1, DOFgh1   = compute_alpha_gh(nwavegh1, nfluxgh1, nweightgh1,  waveRange(nwavegh1))
            AMPa2g1, CENTERa2g1, SIGMAa2g1, AMPb2g1, CENTERb2g1, SIGMAb2g1, SCALE2g1, ALPHA2g1 ,CHISQ2g1, DOF2g1   = compute_alpha_2g(nwave2g1, nflux2g1, nweight2g1,  waveRange(nwave2g1))
            AMPv1, CENTERv1, SIGMALv1, SIGMAGv1,  SCALEv1, ALPHAv1, CHISQv1, DOFv1   = compute_alpha_voigt(nwavev1, nfluxv1, nweightv1,  waveRange(nwavev1))
            AMPgh2, CENTERgh2, SIGMAgh2, SKEWgh2, KURTgh2, SCALEgh2, ALPHAgh2, CHISQgh2, DOFgh2   = compute_alpha_gh(nwavegh2, nfluxgh2, nweightgh2,  waveRange(nwavegh2))
            AMPa2g2, CENTERa2g2, SIGMAa2g2, AMPb2g2, CENTERb2g2, SIGMAb2g2, SCALE2g2, ALPHA2g2 ,CHISQ2g2, DOF2g2   = compute_alpha_2g(nwave2g2, nflux2g2, nweight2g2,  waveRange(nwave2g2))
            AMPv2, CENTERv2, SIGMALv2, SIGMAGv2,  SCALEv2, ALPHAv2, CHISQv2, DOFv2   = compute_alpha_voigt(nwavev2, nfluxv2, nweightv2,  waveRange(nwavev2))
            AMPgh3, CENTERgh3, SIGMAgh3, SKEWgh3, KURTgh3, SCALEgh3, ALPHAgh3, CHISQgh3, DOFgh3   = compute_alpha_gh(nwavegh3, nfluxgh3, nweightgh3,  waveRange(nwavegh3))
            AMPa2g3, CENTERa2g3, SIGMAa2g3, AMPb2g3, CENTERb2g3, SIGMAb2g3, SCALE2g3, ALPHA2g3 ,CHISQ2g3, DOF2g3   = compute_alpha_2g(nwave2g3, nflux2g3, nweight2g3,  waveRange(nwave2g3))
            AMPv3, CENTERv3, SIGMALv3, SIGMAGv3,  SCALEv3, ALPHAv3, CHISQv3, DOFv3   = compute_alpha_voigt(nwavev3, nfluxv3, nweightv3,  waveRange(nwavev3))



    fig,((ax1,rax1),(ax2,rax2),(ax3,rax3))=plt.subplots(3,2,figsize=(20,10))
    ax1.plot(wave1,flux1,label=str(df['pmf1'][i]))
    ax1.plot(wave1,weight1,alpha=0.2)
    ax1.plot(wave1[weight1 > 0],flux1[weight1>0],'.')
    ax2.plot(wave2,flux2,label=str(df['pmf2'][i]))
    ax2.plot(wave2,weight2,alpha=0.2)
    ax2.plot(wave2[weight2 > 0],flux2[weight2>0],'.')

    ax3.plot(wave3,flux3,label=str(df['pmf3'][i]))
    ax3.plot(wave3,weight3,alpha=0.2)
    ax3.plot(wave3[weight3 > 0],flux3[weight3>0],'.')

    #ax1.plot(cwave1,cflux1,'--')
    pgh1=ax1.plot(wave1,myGaussHermite(wave1,AMPgh1, CENTERgh1, SIGMAgh1, SKEWgh1, KURTgh1, SCALEgh1, ALPHAgh1),':',color='red',lw=3,label='GaussHermite+Power law')
    pgh2=ax2.plot(wave2,myGaussHermite(wave2,AMPgh2, CENTERgh2, SIGMAgh2, SKEWgh2, KURTgh2, SCALEgh2, ALPHAgh2),':',color='red',lw=3,label='GaussHermite+Power law')
    pgh3=ax3.plot(wave3,myGaussHermite(wave3,AMPgh3, CENTERgh3, SIGMAgh3, SKEWgh3, KURTgh3, SCALEgh3, ALPHAgh3),':',color='red',lw=3,label='GaussHermite+Power law')
    
    p2g1=ax1.plot(wave1,myDoubleGauss(wave1,AMPa2g1, CENTERa2g1, SIGMAa2g1, AMPb2g1, CENTERb2g1, SIGMAb2g1, SCALE2g1, ALPHA2g1 ),':',color='blue',lw=3,label='Double Gaussian + Power law')
    p2g2=ax2.plot(wave2,myDoubleGauss(wave2,AMPa2g2, CENTERa2g2, SIGMAa2g2, AMPb2g2, CENTERb2g2, SIGMAb2g2, SCALE2g2, ALPHA2g2 ),':',color='blue',lw=3,label='Double Gaussian + Power law')
    p2g3=ax3.plot(wave1,myDoubleGauss(wave3,AMPa2g3, CENTERa2g3, SIGMAa2g3, AMPb2g3, CENTERb2g3, SIGMAb2g3, SCALE2g3, ALPHA2g3 ),':',color='blue',lw=3,label='Double Gaussian + Power law')
    
    p2v1 = ax1.plot(wave1,myVoigt(wave1,AMPv1, CENTERv1, SIGMALv1, SIGMAGv1,  SCALEv1, ALPHAv1),':',color='orange',lw=3,label='Voigt+ Power law')
    p2v2 = ax2.plot(wave2,myVoigt(wave2,AMPv2, CENTERv2, SIGMALv2, SIGMAGv2,  SCALEv2, ALPHAv2),':',color='orange',lw=3,label='Voigt+ Power law')
    p2v3 = ax3.plot(wave3,myVoigt(wave3,AMPv3, CENTERv3, SIGMALv3, SIGMAGv3,  SCALEv3, ALPHAv3),':',color='orange',lw=3,label='Voigt+ Power law')
    ax1.axhline(cut1,ls=':')
    ax2.axhline(cut1,ls=':')
    ax3.axhline(cut1,ls=':')
    string1gh = 'AMP: {0:4.3f}   CENTER: {1:4.3f} RCHIS:{2:4.3f}'.format(AMPgh1,CENTERgh1,(CHISQgh1/DOFgh1))
    string12g = 'AMP: {0:4.3f}   CENTER: {1:4.3f} RCHIS:{2:4.3f}'.format(AMPa2g1,CENTERa2g1,(CHISQ2g1/ DOF2g1))
    string2gh = 'AMP: {0:4.3f}   CENTER: {1:4.3f} RCHIS:{2:4.3f}'.format(AMPgh2,CENTERgh2,(CHISQgh2/ DOFgh2))
    string22g = 'AMP: {0:4.3f}   CENTER: {1:4.3f} RCHIS:{2:4.3f}'.format(AMPa2g2,CENTERa2g2,(CHISQ2g2/ DOF2g2))
    string3gh = 'AMP: {0:4.3f}   CENTER: {1:4.3f} RCHIS:{2:4.3f}'.format(AMPgh3,CENTERgh3,(CHISQgh3/ DOFgh3))
    string32g = 'AMP: {0:4.3f}   CENTER: {1:4.3f} RCHIS:{2:4.3f}'.format(AMPa2g3,CENTERa2g3,(CHISQ2g3/ DOF2g3))
    ax1.set_xlim(np.min(wave1),np.max(wave1))
    ax2.set_xlim(np.min(wave1),np.max(wave1))
    ax3.set_xlim(np.min(wave1),np.max(wave1))
    xlim=ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ylim2 = ax1.get_ylim()
    ylim3 = ax1.get_ylim()
    ax1.set_ylim(-0.05,4)
    ax2.set_ylim(-0.05,4)
    ax3.set_ylim(-0.05,4)
    ax1.text(xlim[1]-0.15*(xlim[1] - xlim[0]),ylim1[0] + 0.1*(ylim1[1] - ylim1[0]),str(df['pmf1'][i]))
    ax2.text(xlim[1]-0.15*(xlim[1] - xlim[0]),ylim1[0] + 0.1*(ylim1[1] - ylim1[0]),str(df['pmf2'][i]))
    ax3.text(xlim[1]-0.15*(xlim[1] - xlim[0]),ylim1[0] + 0.1*(ylim1[1] - ylim1[0]),str(df['pmf3'][i]))
    #ax1.text(xlim[0]+0.1*(xlim[1] - xlim[0]),ylim1[1] - 0.2*(ylim1[1] - ylim1[0]),string12g)
    #ax2.text(xlim[0]+0.1*(xlim[1] - xlim[0]),ylim2[1] - 0.1*(ylim1[1] - ylim2[0]),string2gh)
    #ax2.text(xlim[0]+0.1*(xlim[1] - xlim[0]),ylim2[1] - 0.2*(ylim1[1] - ylim2[0]),string22g)
    #ax3.text(xlim[0]+0.1*(xlim[1] - xlim[0]),ylim3[1] - 0.1*(ylim1[1] - ylim3[0]),string3gh)
    #ax3.text(xlim[0]+0.1*(xlim[1] - xlim[0]),ylim3[1] - 0.2*(ylim1[1] - ylim2[0]),string32g)
    ax1.legend([pgh1,p2g1,p2v1],['Gaus-Hermite','2-Gaus','Voigt'],loc=3)
    ax1.annotate('Gauss-Hermite:\nAmp = %.2f\nCenter = %.2f\n$\sigma$ = %.2f\nH3 = %.2f\nH4 = %.2f' \
        %(AMPgh1, CENTERgh1, SIGMAgh1, SKEWgh1, KURTgh1),xy=(.05,.95), \
        xycoords='axes fraction',ha="left", va="top", \
    bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    ax1.annotate('Double Gaussian:\nAmp$_1$ = %.2f\nAmp$_2$ = %.2f\nCenter$_1$ = %.2f\nCenter$_2$ = %.2f\n$\sigma_1$ = %.2f\n$\sigma_2$ = %.2f' \
        %(AMPa2g1, AMPb2g1, CENTERa2g1, CENTERb2g1, SIGMAa2g1, SIGMAb2g1),xy=(.95,.95), \
        xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    ax1.annotate('Voigt:\nAmp$_1$ = %.2f\nCenter = %.2f\n$\sigma_l$ = %.2f\n$\sigma_g$ = %.2f' \
        %(AMPv1, CENTERv1, SIGMALv1, SIGMAGv1),xy=(.35,.95), \
        xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)

    
    ax2.legend([pgh2,p2g2,p2v2],['Gaus-Hermite','2-Gaus','Voigt'],prop={'size':8},loc='center left')
    ax2.annotate('Gauss-Hermite:\nAmp = %.2f\nCenter = %.2f\n$\sigma$ = %.2f\nH3 = %.2f\nH4 = %.2f' \
        %(AMPgh2, CENTERgh2, SIGMAgh2, SKEWgh2, KURTgh2),xy=(.05,.95), \
        xycoords='axes fraction',ha="left", va="top", \
    bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    ax2.annotate('Double Gaussian:\nAmp$_1$ = %.2f\nAmp$_2$ = %.2f\nCenter$_1$ = %.2f\nCenter$_2$ = %.2f\n$\sigma_1$ = %.2f\n$\sigma_2$ = %.2f' \
        %(AMPa2g2, AMPb2g2, CENTERa2g2, CENTERb2g2, SIGMAa2g2, SIGMAb2g2),xy=(.95,.95), \
        xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    ax2.annotate('Voigt:\nAmp$_1$ = %.2f\nCenter = %.2f\n$\sigma_l$ = %.2f\n$\sigma_g$ = %.2f' \
        %(AMPv2, CENTERv2, SIGMALv2, SIGMAGv2),xy=(.35,.95), \
        xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)


    ax3.legend([pgh3,p2g3,p2v3],['Gaus-Hermite','2-Gaus','Voigt'],prop={'size':8},loc='center left')
    ax3.annotate('Gauss-Hermite:\nAmp = %.2f\nCenter = %.2f\n$\sigma$ = %.2f\nH3 = %.2f\nH4 = %.2f' \
        %(AMPgh3, CENTERgh3, SIGMAgh3, SKEWgh3, KURTgh3),xy=(.05,.95), \
        xycoords='axes fraction',ha="left", va="top", \
    bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    ax3.annotate('Double Gaussian:\nAmp$_1$ = %.2f\nAmp$_2$ = %.2f\nCenter$_1$ = %.2f\nCenter$_2$ = %.2f\n$\sigma_1$ = %.2f\n$\sigma_2$ = %.2f' \
        %(AMPa2g3, AMPb2g3, CENTERa2g3, CENTERb2g3, SIGMAa2g3, SIGMAb2g3),xy=(.95,.95), \
        xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    ax3.annotate('Voigt:\nAmp$_1$ = %.2f\nCenter = %.2f\n$\sigma_l$ = %.2f\n$\sigma_g$ = %.2f' \
        %(AMPv3, CENTERv3, SIGMALv3, SIGMAGv3),xy=(.35,.95), \
        xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)

    #Save the spectra
    filenamegh1 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf1'][i]+'_Em_gh.txt'
    filenamegh2 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf2'][i]+'_Em_gh.txt'
    filenamegh3 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf3'][i]+'_Em_gh.txt'
    
    filename2g1 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf1'][i]+'_Em_2g.txt'
    filename2g2 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf2'][i]+'_Em_2g.txt'
    filename2g3 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf3'][i]+'_Em_2g.txt'

    filenamev1 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf1'][i]+'_Em_v.txt'
    filenamev2 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf2'][i]+'_Em_v.txt'
    filenamev3 = 'EmissionLines_Norm_Spectra/NormSpec_'+df['pmf3'][i]+'_Em_v.txt'


    # Residuals & Weights
    resgh1 = flux1 / myGaussHermite(wave1,AMPgh1, CENTERgh1, SIGMAgh1, SKEWgh1, KURTgh1, SCALEgh1, ALPHAgh1)
    resgh2 = flux2 / myGaussHermite(wave2,AMPgh2, CENTERgh2, SIGMAgh2, SKEWgh2, KURTgh2, SCALEgh2, ALPHAgh2)
    resgh3 = flux3 / myGaussHermite(wave3,AMPgh3, CENTERgh3, SIGMAgh3, SKEWgh3, KURTgh3, SCALEgh3, ALPHAgh3)
    
    wresgh1 = weight1 / myGaussHermite(wave1,AMPgh1, CENTERgh1, SIGMAgh1, SKEWgh1, KURTgh1, SCALEgh1, ALPHAgh1)
    wresgh2 = weight2 / myGaussHermite(wave2,AMPgh2, CENTERgh2, SIGMAgh2, SKEWgh2, KURTgh2, SCALEgh2, ALPHAgh2)
    wresgh3 = weight3 / myGaussHermite(wave3,AMPgh3, CENTERgh3, SIGMAgh3, SKEWgh3, KURTgh3, SCALEgh3, ALPHAgh3)

    res2g1 = flux1 / myDoubleGauss(wave1,AMPa2g1, CENTERa2g1, SIGMAa2g1, AMPb2g1, CENTERb2g1, SIGMAb2g1, SCALE2g1, ALPHA2g1 )
    res2g2 = flux2 / myDoubleGauss(wave2,AMPa2g2, CENTERa2g2, SIGMAa2g2, AMPb2g2, CENTERb2g2, SIGMAb2g2, SCALE2g2, ALPHA2g2 )
    res2g3 = flux3 / myDoubleGauss(wave3,AMPa2g3, CENTERa2g3, SIGMAa2g3, AMPb2g3, CENTERb2g3, SIGMAb2g3, SCALE2g3, ALPHA2g3 )

    wres2g1 = weight1 / myDoubleGauss(wave1,AMPa2g1, CENTERa2g1, SIGMAa2g1, AMPb2g1, CENTERb2g1, SIGMAb2g1, SCALE2g1, ALPHA2g1 )
    wres2g2 = weight2 / myDoubleGauss(wave2,AMPa2g2, CENTERa2g2, SIGMAa2g2, AMPb2g2, CENTERb2g2, SIGMAb2g2, SCALE2g2, ALPHA2g2 )
    wres2g3 = weight3 / myDoubleGauss(wave3,AMPa2g3, CENTERa2g3, SIGMAa2g3, AMPb2g3, CENTERb2g3, SIGMAb2g3, SCALE2g3, ALPHA2g3 )

    resv1 = flux1 / myVoigt(wave1,AMPv1, CENTERv1, SIGMALv1, SIGMAGv1,  SCALEv1, ALPHAv1)
    resv2 = flux2 / myVoigt(wave2,AMPv2, CENTERv2, SIGMALv2, SIGMAGv2,  SCALEv2, ALPHAv2)
    resv3 = flux3 / myVoigt(wave3,AMPv3, CENTERv3, SIGMALv3, SIGMAGv3,  SCALEv3, ALPHAv3)

    wresv1 = weight1 / myVoigt(wave1,AMPv1, CENTERv1, SIGMALv1, SIGMAGv1,  SCALEv1, ALPHAv1)
    wresv2 = weight2 / myVoigt(wave2,AMPv2, CENTERv2, SIGMALv2, SIGMAGv2,  SCALEv2, ALPHAv2)
    wresv3 = weight3 / myVoigt(wave3,AMPv3, CENTERv3, SIGMALv3, SIGMAGv3,  SCALEv3, ALPHAv3)

    #Save Begins
    np.savetxt(filenamegh1,zip(wave1,resgh1,wresgh1,mask1), fmt='%10.5f')
    np.savetxt(filenamegh2,zip(wave2,resgh2,wresgh2,mask2), fmt='%10.5f')
    np.savetxt(filenamegh3,zip(wave3,resgh3,wresgh3,mask3), fmt='%10.5f')

    np.savetxt(filename2g1,zip(wave1,res2g1,wres2g1,mask1), fmt='%10.5f')
    np.savetxt(filename2g2,zip(wave2,res2g2,wres2g2,mask2), fmt='%10.5f')
    np.savetxt(filename2g3,zip(wave3,res2g3,wres2g3,mask3), fmt='%10.5f')

    np.savetxt(filenamev1,zip(wave1,resv1,wresv1,mask1), fmt='%10.5f')
    np.savetxt(filenamev2,zip(wave2,resv2,wresv2,mask2), fmt='%10.5f')
    np.savetxt(filenamev3,zip(wave3,resv3,wresv3,mask3), fmt='%10.5f')

    #Plot Begins
    presgh1,pres2g1,presv1,=rax1.plot(wave1,resgh1,'k--',wave1,res2g1,'r-',wave1,resv1,'b:',alpha=0.7)
    presgh2,pres2g2,presv2,=rax2.plot(wave2,resgh2,'k--',wave2,res2g2,'r-',wave2,resv2,'b:',alpha=0.7)
    presgh3,pres2g3,presv3,=rax3.plot(wave3,resgh3,'k--',wave3,res2g3,'r-',wave3,resv3,'b:',alpha=0.7)

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

