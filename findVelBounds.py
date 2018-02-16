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
#from dustmaps.sfd import SFDQuery
#from specutils import extinction  
import scipy
from scipy.ndimage.filters import convolve
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit 
from matplotlib.backends.backend_pdf import PdfPages
#from lmfit import minimize, Parameters
from astropy.modeling.models import Voigt1D
from collections import Iterable

def flatten(items):
    """Yield items from any nested iterable; see REF."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            return  flatten(x)
        else:
            return x

def velocity(wave,ion_wave,z):
    #print wave
    #print z,ion_wave
    c = 299792.0# in km/s
    vel =np.zeros(len(wave))
    zabs = np.zeros(len(wave))
    for i in range(len(wave)):
        #print i,wave[i],zabs[i],vel[i]
        zabs[i] = (wave[i]/ion_wave) -1.0
        vel[i] = -((((1+z)/(1+zabs[i]))**2-1.0)/(((1+z)/(1+zabs[i]))**2+1))*c
    return vel



def balBounds(vel,flux,threshold=0.9,min_vel=2000):
  start = -1
  contigous_segments = []
  for idx,x in enumerate(zip(vel,flux)):
    if start < 0 and x[1] <= threshold:
      start = idx
      #print 'start', start
    elif start >= 0 and x[1] >= threshold:
      dur = idx-start
      veldiff = vel[idx] - vel[start]
      #print 'velocity difference: ',dur, veldiff
      #Multiply by -1 to take care of blueshifted velocities
      if veldiff >= min_vel:
        contigous_segments.append((vel[start],vel[start+dur]))
      start = -1
  return contigous_segments


def mergeVelocityBounds(bounds1,bounds2,bounds3):

    masterbounds = bounds1+bounds2+bounds3
    sortedbounds = sorted(masterbounds, key=lambda t: t[0])
    print 'MasterBounds',masterbounds
    print 'SortedBounds',sortedbounds
    merged = []

    for higher in sortedbounds:
        print 'Higher',higher
        print 'M',merged
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            #print 'To set the merging of adjacent BALs',higher[0] - lower[1]
            # this condition to merge close by BALs
            if higher[0] > lower[1]:
                if higher[0] - lower[1] <= 50:
                
                    merged[-1] = (lower[0], higher[1])  # replace by merged interval
                else:
                    merged.append(higher)
            elif higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1] = (lower[0], upper_bound)  # replace by merged interval
            else:
                merged.append(higher)
    print 'Merged',merged
    return merged
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

def topHat(mvel1,merged):
    #Plot a top hat function
    mask1 = np.ones(len(mvel1))*(1==0)
    #print 'VelMerged',mvel1
    indices = list()
    for m in merged:
        #print m[0],m[1]
        ix = np.where((mvel1 >= m[0]) & (mvel1 <=m[1]))[0]
        #print len(ix)
        if len(ix)>0:
            indices.append(ix)
    #print len(indices)
    indices = np.array(indices)
    findices = [y for x in indices for y in x]
    #print findices,
    #print len(findices)
    if len(findices) >0:
        mask1[findices] = True

    return mask1

c_light=299792.0 
cw=1549.0
df = np.genfromtxt('tdss_allmatches_crop_edit.dat',names=['ra','dec','z','pmf1','pmf2','pmf3'],dtype=(float,float,float,'|S15','|S15','|S15'))
pp = PdfPages('BALcomponents_search_I.pdf')
fx=open('BALcomponents_output.txt','w')
#for i in range(len(df['pmf1'])):
for i in range(len(df)):
#for i in range(5):

    filename1 = 'Normspec_'+df['pmf1'][i]+'_cEm_2g.txt' 
    filename2 = 'Normspec_'+df['pmf2'][i]+'_cEm_2g.txt' 
    filename3 = 'Normspec_'+df['pmf3'][i]+'_cEm_2g.txt' 
    if ((not os.path.isfile('EmissionLines_Norm_Spectra/Normspec_'+df['pmf1'][i]+'_cEm_2g.txt')) | (not os.path.isfile('EmissionLines_Norm_Spectra/Normspec_'+df['pmf2'][i]+'_cEm_2g.txt')) | ( not os.path.isfile('EmissionLines_Norm_Spectra/Normspec_'+df['pmf3'][i]+'_cEm_2g.txt'))):
        continue
    else:
        data1 = np.loadtxt('EmissionLines_Norm_Spectra/Normspec_'+df['pmf1'][i]+'_cEm_2g.txt')
        data2 = np.loadtxt('EmissionLines_Norm_Spectra/Normspec_'+df['pmf2'][i]+'_cEm_2g.txt')
        data3 = np.loadtxt('EmissionLines_Norm_Spectra/Normspec_'+df['pmf3'][i]+'_cEm_2g.txt')
    wave1 = data1.T[0] ;    bflux1 = data1.T[1] ; weight1 =  data1.T[2]#; mask1 = data1.T[3]
    wave2 = data2.T[0] ;    bflux2 = data2.T[1] ; weight2 =  data2.T[2]#; mask2 = data2.T[3]
    wave3 = data3.T[0] ;    bflux3 = data3.T[1] ; weight3 =  data3.T[2]#; mask3 = data3.T[3]

    if ((len(bflux1[~np.isnan(bflux1)]) < 1) | (len(bflux2[~np.isnan(bflux2)]) < 1) | (len(bflux3[~np.isnan(bflux3)]) < 1)  ) :
        filename1 = 'Normspec_'+df['pmf1'][i]+'_cEm_v.txt' 
        filename2 = 'Normspec_'+df['pmf2'][i]+'_cEm_v.txt' 
        filename3 = 'Normspec_'+df['pmf3'][i]+'_cEm_v.txt' 

        data1 = np.loadtxt('EmissionLines_Norm_Spectra/Normspec_'+df['pmf1'][i]+'_cEm_v.txt')
        data2 = np.loadtxt('EmissionLines_Norm_Spectra/Normspec_'+df['pmf2'][i]+'_cEm_v.txt')
        data3 = np.loadtxt('EmissionLines_Norm_Spectra/Normspec_'+df['pmf3'][i]+'_cEm_v.txt')
        wave1 = data1.T[0] ;    bflux1 = data1.T[1] ; weight1 =  data1.T[2]#; mask1 = data1.T[3]
        wave2 = data2.T[0] ;    bflux2 = data2.T[1] ; weight2 =  data2.T[2]#; mask2 = data2.T[3]
        wave3 = data3.T[0] ;    bflux3 = data3.T[1] ; weight3 =  data3.T[2]#; mask3 = data3.T[3]

    
    #Work on smoothed spectrum to get rid og high frequency noise
    flux1 = savitzky_golay(bflux1, 5, 2)
    flux2 = savitzky_golay(bflux2, 5, 2)
    flux3 = savitzky_golay(bflux3, 5, 2)
    print len(wave1),len(flux1),len(weight1)    
    
    #Convert wavelengths to velocities
    vel1 = velocity(wave1, cw, 0.0) ; vel2 = velocity(wave2, cw, 0.0) ; vel3 = velocity(wave3, cw, 0.0)
    bounds1 = balBounds(vel1,flux1) ; bounds2 = balBounds(vel2,flux2); bounds3 = balBounds(vel3, flux3)    
    print 'bounds1', bounds1
    print 'bounds2', bounds2
    print 'bounds3', bounds3
    
    #Merge the overlapping regions
    merged = mergeVelocityBounds(bounds1,bounds2,bounds3)
    ntroughs = len(merged)
    
    #Check if each component is present in atleast 2 epochs
    #topHat function returns a mask with ones inside the absorption
    mask1 = topHat(vel1,bounds1) ; mask2 = topHat(vel1,bounds2) ; mask3 = topHat(vel1,bounds3)
    for kk,mm in enumerate(merged):
        #print mm
        #print topHat(vel1,[mm])
        #print mask1
        track_absorption = 0
        if np.sum( mask1.any() and topHat(vel1,[mm])) > 0:
            track_absorption += 1
        if np.sum(mask2.any() and topHat(vel1,[mm])) > 0:
            track_absorption += 1
        if np.sum(mask3.any() and topHat(vel1,[mm])) > 0:
            track_absorption += 1
        if track_absorption < 2:
            merged.pop(kk)
    
    maskall = topHat(vel1,merged)
    #Get the min max of each bounds
    ubounds = []
    if (len(bounds1) == len(bounds2) == len(bounds3)):
      for k in range(len(bounds1)):

        uboundsl,uboundsu = min(bounds1[k][0],bounds2[k][0],bounds3[k][0]),max(bounds1[k][1],bounds2[k][1],bounds3[k][1])
        ubounds.append((uboundsl,uboundsu))
    else:
        print 'Manually adjust ',df[i]
    
    #Plot the spectra and check the bounds
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True, sharey=True,figsize=(15,8))
    ax1.plot(vel1,bflux1,color='black',alpha=0.3,label=str(df['pmf1'][i]))
    ax1.plot(vel1,flux1,color='magenta',alpha=0.3)
    ax1.plot(vel1,maskall*0.5,color='green')
    for bb in bounds1:
        ax1.axvline(bb[0],ls='--',color='blue',lw=1)
        ax1.axvline(bb[1],ls='--',color='red',lw=1)
    ax2.plot(vel2,bflux2,color='black',alpha=0.3,label=str(df['pmf2'][i]))
    ax2.plot(vel2,flux2,color='magenta',alpha=0.3)
    ax2.plot(vel1,maskall*0.5,color='green')
    for bb in bounds2:
        ax2.axvline(bb[0],ls='--',color='blue',lw=1)
        ax2.axvline(bb[1],ls='--',color='red',lw=1)
    ax3.plot(vel3,bflux3,color='black',alpha=0.3,label=str(df['pmf3'][i]))
    ax3.plot(vel3,flux3,color='magenta',alpha=0.3)
    ax3.plot(vel1,maskall*0.5,color='green')
    for bb in bounds3:
        ax3.axvline(bb[0],ls='--',color='blue',lw=1)
        ax3.axvline(bb[1],ls='--',color='red',lw=1)
    for ubb in ubounds:
        #print 'final bounds',ubounds
        ax1.axvline(ubb[0],ls='-',color='blue',lw=1)
        ax2.axvline(ubb[0],ls='-',color='blue',lw=1)
        ax3.axvline(ubb[0],ls='-',color='blue',lw=1)
        ax1.axvline(ubb[1],ls='-',color='red',lw=1)
        ax2.axvline(ubb[1],ls='-',color='red',lw=1)
        ax3.axvline(ubb[1],ls='-',color='red',lw=1)

    #ax1.set_xlabel(r'Velocity km s$^{-1}$')
    #ax2.set_xlabel(r'Velocity km s$^{-1}$')
    ax3.set_xlabel(r'Velocity km s$^{-1}$')
    
    ax1.axhline(1.0,ls='--',color='orange')
    ax2.axhline(1.0,ls='--',color='orange')
    ax3.axhline(1.0,ls='--',color='orange')

    ax1.axhline(0.9,ls='--',color='gold')
    ax2.axhline(0.9,ls='--',color='gold')
    ax3.axhline(0.9,ls='--',color='gold')

    ax1.axvspan(-30000,1000,color='grey',alpha=0.1)
    ax2.axvspan(-30000,1000,color='grey',alpha=0.1)
    ax3.axvspan(-30000,1000,color='grey',alpha=0.1)
    ax1.set_ylim(0,2)
    ax2.set_ylim(0,2)
    ax3.set_ylim(0,2)
    #ax1.set_ylabel(r'Normalized Flux')
    #ax2.set_ylabel(r'Normalized Flux')
    ax3.set_ylabel(r'Normalized Flux')
    ax1.legend(loc=1)
    ax2.legend(loc=1)
    ax3.legend(loc=1)
    
    for j in range(len(merged)):
        merge = merged[j]
        print>>fx,'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(i,filename1,merge[1],merge[0],0,ntroughs, 2000.0000,2000.0000,10.0,10.0)
        print>>fx,'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(i,filename2,merge[1],merge[0],0,ntroughs, 2000.0000,2000.0000,10.0,10.0)
        print>>fx,'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(i,filename3,merge[1],merge[0],0,ntroughs, 2000.0000,2000.0000,10.0,10.0)
    #plt.show()
    #plt.clf()
    # Some book keeping
    
    fig.tight_layout()
    fig.savefig(pp,format='pdf')
   # sinp = raw_input('Type Some Key to Continue: ')
pp.close()
fx.close()
