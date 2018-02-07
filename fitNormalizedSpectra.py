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

def gaussfunc_gh(paramsin,x):
    print paramsin
    amp=paramsin['amp'].value
    center=paramsin['center'].value
    sig=paramsin['sig'].value
    c1=-np.sqrt(3); c2=-np.sqrt(6); c3=2/np.sqrt(3); c4=np.sqrt(6)/3; c5=np.sqrt(6)/4
    skew=paramsin['skew'].value
    kurt=paramsin['kurt'].value
    scale = paramsin['scale'].value ; alpha = paramsin['alpha'].value 
 
    gaustot_gh=amp*np.exp(-.5*((x-center)/sig)**2)*(1+skew*(c1*((x-center)/sig)+c3*((x-center)/sig)**3)+kurt*(c5+c2*((x-center)/sig)**2+c4*((x-center)/sig)**4))+scale*x**alpha
    return gaustot_gh


def gaussfunc_2g(paramsin,x):
    amp1=paramsin['amp1'].value; amp2=paramsin['amp2'].value;
    center1=paramsin['center1'].value; center2=paramsin['center2'].value;
    sig1=paramsin['sig1'].value; sig2=paramsin['sig2'].value;
    scale = paramsin['scale'].value ; alpha = paramsin['alpha'].value 
    gaus1=amp1*np.exp(-.5*((x-center1)/sig1)**2)
    gaus2=amp2*np.exp(-.5*((x-center2)/sig2)**2)
    gaustot_2g=(gaus1+gaus2+scale*x**alpha)
    return gaustot_2g


def PLot(vels,stackspec , fit_gh, fit_2g, pars_gh, pars_2g, resid_gh, resid_2g):
    fig3=plt.figure(3)
    f1=fig3.add_axes((.1,.3,.8,.6))
        #xstart, ystart, xwidth, yheight --> units are fraction of the image from bottom left
    
    plt.plot(vels,stackspec,'k.')
    pgh,=plt.plot(vels,fit_gh,'b')
    p2g,=plt.plot(vels,fit_2g,'r')
    f1.set_xticklabels([]) #We will plot the residuals below, so no x-ticks on this plot
    plt.title('Multiple Gaussian Fit Example')
    plt.ylabel('Amplitude (Some Units)')
    f1.legend([pgh,p2g],['Gaus-Hermite','2-Gaus'],prop={'size':10},loc='center left')
     
    from matplotlib.ticker import MaxNLocator
    plt.gca().yaxis.set_major_locator(MaxNLocator(prune='lower')) #Removes lowest ytick label
    
    f1.annotate('Gauss-Hermite:\nAmp = %.2f\nCenter = %.2f\n$\sigma$ = %.2f\nH3 = %.2f\nH4 = %.2f' \
        %(pars_gh[0],pars_gh[1],pars_gh[2],pars_gh[3],pars_gh[4]),xy=(.05,.95), \
        xycoords='axes fraction',ha="left", va="top", \
    bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    f1.annotate('Double Gaussian:\nAmp$_1$ = %.2f\nAmp$_2$ = %.2f\nCenter$_1$ = %.2f\nCenter$_2$ = %.2f\n$\sigma_1$ = %.2f\n$\sigma_2$ = %.2f' \
        %(pars_2g[0],pars_2g[3],pars_2g[1],pars_2g[4],pars_2g[2],pars_2g[5]),xy=(.95,.95), \
        xycoords='axes fraction',ha="right", va="top", \
        bbox=dict(boxstyle="round", fc='1'),fontsize=10)
    
    f2=fig3.add_axes((.1,.1,.8,.2))
     
    resgh,res2g,=plt.plot(vels,resid_gh,'k--',vels,resid_2g,'k')
     
    plt.ylabel('Residuals')
    plt.xlabel('Velocity (km s$^{-1}$)')
    f2.legend([resgh,res2g],['Gaus-Hermite','2-Gaus'],numpoints=4,prop={'size':9},loc='upper left')
     
    plt.savefig(pp,format='pdf')
    plt.show()
     
    plt.clf()


df = np.genfromtxt('tdss_allmatches_crop_edit.dat',names=['ra','dec','z','pmf1','pmf2','pmf3'],dtype=(float,float,float,'|S15','|S15','|S15'))
wav_range= [(1280,1350),(1700,1800),(1950,2200),(2650,2710),(3010,3700),(3950,4050),(4140,4270)]
pp = PdfPages('ContinuumNormalization_plus_EmissionLineFits.pdf')
#fx=open('ALPHA_AMP_values_V.txt','w')
#fx1=open('Kate_ALPHA_AMP_values_V.txt','w')
#for i in range(len(df['pmf1'])):
#for i in range(len(df)):
for i in range(4):

    print 'Norm_Spectra/Normspec_'+df['pmf1'][i]+'.txt' 
    print 'Norm_Spectra/Normspec_'+df['pmf2'][i]+'.txt' 
    print 'Norm_Spectra/Normspec_'+df['pmf3'][i]+'.txt' 
    if ((not os.path.isfile('Norm_Spectra/Normspec_'+df['pmf1'][i]+'.txt')) | (not os.path.isfile('Norm_Spectra/Normspec_'+df['pmf2'][i]+'.txt')) | ( not os.path.isfile('Norm_Spectra/Normspec_'+df['pmf3'][i]+'.txt'))):
        continue
    else:
        data1 = np.loadtxt('Norm_Spectra/Normspec_'+df['pmf1'][i]+'.txt')
        data2 = np.loadtxt('Norm_Spectra/Normspec_'+df['pmf2'][i]+'.txt')
        data3 = np.loadtxt('Norm_Spectra/Normspec_'+df['pmf3'][i]+'.txt')
    wave1 = data1.T[0] ;    flux1 = data1.T[1] ; weight1 =  data1.T[2]
    wave2 = data2.T[0] ;    flux2 = data2.T[1] ; weight2 =  data2.T[2]
    wave3 = data3.T[0] ;    flux3 = data3.T[1] ; weight3 =  data3.T[2]
    print len(wave1),len(flux1),len(weight1)
    
    print weight1
    
    p_gh=Parameters()
    p_gh.add('amp',value=np.max(flux1),vary=True);
    p_gh.add('center',value=1550,min=1540,max=1560,vary=True);
    p_gh.add('sig',value=15,min=3,max=10,vary=True);
    p_gh.add('skew',value=0,vary=True,min=None,max=None);
    p_gh.add('kurt',value=0,vary=True,min=None,max=None);
    p_gh.add('scale',value=1,vary=True,min=None,max=None);
    p_gh.add('alpha',value=-1.5,vary=True,min=-3,max=3);
 
    p_2g=Parameters()
    p_2g.add('amp1',value=np.max(flux1)/2.,min=.1*np.max(flux1),max=np.max(flux1),vary=True);
    p_2g.add('center1',value=1550,min=1540,max=1560,vary=True);
    p_2g.add('sig1',value=15,min=3,max=10,vary=True);
    p_2g.add('amp2',value=np.max(flux1)/2.,min=.1*np.max(flux1),max=np.max(flux1),vary=True);
    p_2g.add('center2',value=1545,min=1540,max=1560,vary=True);
    p_2g.add('sig2',value=15,min=3,max=10,vary=True);
    p_2g.add('scale',value=1,vary=True,min=None,max=None);
    p_2g.add('alpha',value=-1.5,vary=True,min=-3,max=3);

    gausserr_gh = lambda p,x,y: gaussfunc_gh(p,x)-y
    gausserr_2g = lambda p,x,y: gaussfunc_2g(p,x)-y

    fitout_gh=minimize(gausserr_gh,p_gh,args=(wave1,flux1))
    fitout_2g=minimize(gausserr_2g,p_2g,args=(wave1,flux1))
    
    pars_gh=[p_gh['amp'].value,p_gh['center'].value,p_gh['sig'].value,p_gh['skew'].value,p_gh['kurt'].value]
    pars_2g=[p_2g['amp1'].value,p_2g['center1'].value,p_2g['sig1'].value,p_2g['amp2'].value,p_2g['center2'].value,p_2g['sig2'].value]

    fit_gh=gaussfunc_gh(p_gh,wave1)
    fit_2g=gaussfunc_2g(p_2g,wave1)
    resid_gh=fit_gh- flux1
    resid_2g=fit_2g- flux1

    print('Fitted Parameters (Gaus+Hermite):\nAmp = %.2f , Center = %.2f , Disp = %.2f\nSkew = %.2f , Kurt = %.2f' \
%(pars_gh[0],pars_gh[1],pars_gh[2],pars_gh[3],pars_gh[4]))
 
    print('Fitted Parameters (Double Gaussian):\nAmp1 = %.2f , Center1 = %.2f , Sig1 = %.2f\nAmp2 = %.2f , Center2 = %.2f , Sig2 = %.2f' \
%(pars_2g[0],pars_2g[1],pars_2g[2],pars_2g[3],pars_2g[4],pars_2g[5]))
    PLot(wave1,flux1,fit_gh, fit_2g, pars_gh, pars_2g, resid_gh, resid_2g)
