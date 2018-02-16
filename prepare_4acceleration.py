#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
import argparse
import os
from astropy import io
from astropy.io import ascii
from astropy.table import Table
from scipy.ndimage.filters import convolve
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy import optimize 
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
import numpy as np
from astropy.io import fits
import scipy as sp
import matplotlib.pyplot as plt
from pydl.pyspherematch import *
from pydl.pydl.pydlutils import yanny
#from pydl.pydl.pydlutils.spheregroup import *
import os
from astropy.time import Time

def choose_3epochs(all_epochs):
    sortedpmf = sorted(all_epochs, key=lambda t: t[1])
    nepochs = len(sortedpmf)
    sdsspmf = []; bosspmf=[];ebosspmf = []
    for item in sortedpmf:
        if item[1] <= 55560:
            sdsspmf.append(item)
        elif ((item[1] > 55560) & (item[1] <= 56665)) :
            bosspmf.append(item)
        else :
            ebosspmf.append(item)
    print 'sdsspmf', sdsspmf
    print 'bosspmf', bosspmf
    print 'ebosspmf', ebosspmf
    if len(sdsspmf) > 0 :
        pickedsdss = sdsspmf[0]
    if len(bosspmf) > 0 :
        pickedboss = bosspmf[-1]
    if len(ebosspmf) > 0 :
        pickedeboss = ebosspmf[-1]
    if ((len(sdsspmf) == 0) & (len(bosspmf) >1) ):
        pickedsdss = bosspmf[0]
    if ((len(sdsspmf) == 0) & (len(bosspmf) <=1) & (len(ebosspmf) >1)) :
        pickedsdss = ebosspmf[0]
    if ((len(sdsspmf) == 0) & (len(bosspmf) ==0) & (len(ebosspmf) >2)) :
        pickedeboss = ebosspmf[-1]
        pickedsdss = ebosspmf[0]
        pickedboss = ebosspmf[-2]

    if ((len(bosspmf) ==  0) & (len(sdsspmf) >1)):
        pickedboss = sdsspmf[-1]
    if ((len(bosspmf) ==  0) & (len(sdsspmf) ==1) &(len(ebosspmf) >1)):
        pickedboss = ebosspmf[0]
    if ((len(ebosspmf) ==  0) & (len(bosspmf) >1)):
        pickedeboss = bosspmf[0]
    if ((len(ebosspmf) ==  0) & (len(bosspmf) ==1) &(len(sdsspmf) >1)):
        pickedeboss = sdsspmf[-1]

    return pickedeboss,pickedsdss,pickedboss


targets = open('targets_with3epochs_dr12_spherematch.txt','w')
check  =open('targets_with_lt_2epochs_dr12_spherematch.txt','w')

tolerance_arcsec=1.5
tolerance_deg = tolerance_arcsec/3600.

spAllfile = 'spAll-v5_10_7.fits'
#
data = fits.open('spAll-v5_10_0.fits')[1].data

spAll = fits.open(spAllfile)[1].data

xx=np.where(spAll['EBOSS_TARGET2'] & 2**25)[0]


yy=np.where(data['PROGRAMNAME'] == 'sequels')[0]
sequels = data[yy]
xx1=np.where(sequels['EBOSS_TARGET0'] & 2**35)[0]

sequelsbals = sequels[xx1]

balspAll = spAll[xx]
sequelsra= sequelsbals['RA'] ; sequelsdec=sequelsbals['DEC'];sequelsmjd = sequelsbals['MJD'];sequelsplate=sequelsbals['PLATE']
sequelsfiber = sequelsbals['FIBERID'];sequelsz=sequelsbals['Z']
epra = balspAll['RA'];epdec=balspAll['DEC'];eplate = balspAll['PLATE'];efiber = balspAll['FIBERID']
emjd = balspAll['MJD'];ebossz = balspAll['Z']

#print np.min(sequelsra),np.max(sequelsra)
pra=np.concatenate((epra,sequelsra))
pdec=np.concatenate((epdec,sequelsdec))
mjd=np.concatenate((emjd,sequelsmjd))
plate = np.concatenate((eplate,sequelsplate))
fiber = np.concatenate((efiber,sequelsfiber))
z = np.concatenate((ebossz,sequelsz))
#print len(epra),len(sequelsra),len(pra)
radec = zip(pra,pdec)
unq, unq_inv, unq_cnt = np.unique(pra, return_inverse=True, return_counts=True)
#print len(unq_cnt),unq_cnt
oldlist = [];newlist=[];un_mjd=[];un_plate=[];un_fiber=[]

for k in range(len(radec)):
    if radec[k] not in oldlist:
        oldlist.append(radec[k])
        newlist.append(radec[k])
        un_mjd.append(mjd[k])
        un_plate.append(plate[k])
        un_fiber.append(fiber[k])
    else:
        pass
ura = [];udec=[]
n_epochs=[]

#Readin Hewett & Wild 2010 Redshifts 
hra,hdec,hz,hzerr,first,ahz,zmethod,hplate,hmjd,hfiber = np.loadtxt('/Users/vzm83/Downloads/1045471_Supplementary_Data/mnras0408-2302-SD1.txt',usecols=(1,2,3,4,5,6,7,8,9,10)).T
hplate=[int(x) for x in hplate] ; hmjd=[int(x) for x in hmjd] ; hfiber = [int(x) for x in hfiber]
completera = []; completedec = [];completeredshift = []; complete_epochs = []; completeredshift_tag = []
for j in range(len(newlist)):
    print str(j)+'/'+str(len(newlist)), newlist[j]
    mra=newlist[j][0];mdec=newlist[j][1]
    all_epochs = []
    ep =  np.where((pra == mra) & (pdec == mdec))[0]
    dr14q = fits.open('../DR14QSO/DR14Q_v4_4.fits')[1].data
    drm =  np.where((dr14q['RA'] == mra) & (dr14q['DEC'] == mdec))[0]
    adrm,drm,dds= spherematch(mra,mdec,dr14q['RA'],dr14q['DEC'],tol=tolerance_deg)

    if len(drm) > 0:
        dup_plates = dr14q['PLATE_DUPLICATE'][drm[0]] ;dup_mjds = dr14q['MJD_DUPLICATE'][drm[0]]; dup_fibers = dr14q['FIBERID_DUPLICATE'][drm[0]]
        ind = np.where(dup_plates > 0)[0]
        dup_plates = dup_plates[ind] ; dup_mjds = dup_mjds[ind];dup_fibers = dup_fibers[ind]
       # print 'DUPLICATES: ',dr14q['RA'][drm[0]],dr14q['DEC'][drm[0]],dr14q['N_SPEC'][drm[0]],dup_plates,dup_mjds,dup_fibers
        for n in range(len(drm)):
      #      print 'DR14Q {}\t{}\t{}\t{}\t{}'.format(dr14q['RA'][drm[n]],dr14q['DEC'][drm[n]],dr14q['PLATE'][drm[n]],dr14q['MJD'][drm[n]],dr14q['FIBERID'][drm[n]])
            all_epochs.append((dr14q['PLATE'][drm[n]],dr14q['MJD'][drm[n]],dr14q['FIBERID'][drm[n]]))
            for ii in range(len(dup_plates)):
                if (dup_plates[ii],dup_mjds[ii],dup_fibers[ii]) not in all_epochs :
                    all_epochs.append((dup_plates[ii],dup_mjds[ii],dup_fibers[ii]))
    n_epochs.append(len(ep))
    ura.append(mra);udec.append(mdec)
    for m in range(len(ep)):
     #   print 'SPALL {}\t{}\t{}\t{}\t{}\t{}'.format(pra[ep[m]],pdec[ep[m]],plate[ep[m]],mjd[ep[m]],fiber[ep[m]],len(ep))
        if (plate[ep[m]],mjd[ep[m]],fiber[ep[m]]) not in all_epochs:
            all_epochs.append((plate[ep[m]],mjd[ep[m]],fiber[ep[m]]))
    print 'All_epochs :', all_epochs
    a,b,ds= spherematch(mra,mdec,hra,hdec,tol=tolerance_deg)
    #print a,b
    if len(a) > 0:
        print 'Hewett & Wild',hra[b],hdec[b], hz[b]
        redshift = hz[b]
        redshift_tag = "HW10"
    else:
        redshift = z[[ep[0]]] 
        redshift_tag = "PIPE"
       # print 'No match in Hewett & Wild'
    print 'Redshift: ',redshift, redshift_tag
    completera.append(mra)
    completedec.append(mdec)
    completeredshift.append(redshift)
    completeredshift_tag.append(redshift_tag)
    complete_epochs.append(all_epochs)
    if len(all_epochs) >= 3 :
        pickedeboss,pickedsdss,pickedboss = choose_3epochs(all_epochs)
        formatpickedeboss = '{0:04d}-{1:05d}-{2:04d}'.format(pickedeboss[0],pickedeboss[1],pickedeboss[2])
        formatpickedsdss = '{0:04d}-{1:05d}-{2:04d}'.format(pickedsdss[0],pickedsdss[1],pickedsdss[2])
        formatpickedboss = '{0:04d}-{1:05d}-{2:04d}'.format(pickedboss[0],pickedboss[1],pickedboss[2])
        print>>targets, '{0:10.5f}\t{1:10.5f}\t{2:8.4f}\t{3}\t{4}\t{5}'.format(mra,mdec,redshift[0],formatpickedeboss,formatpickedsdss,formatpickedboss)
    else:
        print>>check, 'Less than Three epochs: {0:10.5f}\t{1:10.5f}\t{2:8.4f}\t{3}'.format(mra,mdec,redshift[0],all_epochs)
    print '--'*51
    #pkpk=raw_input()

np.savez('output_prepare4acceleration_dr14_spherematch.npz',
        ra = completera ,
        dec = completedec ,
        redshift = completeredshift ,
        redshift_tag = completeredshift_tag ,
        pmf = complete_epochs)
#c1 = fits.Column(name='ra', array=np.array(completera), format='K')
#c2 = fits.Column(name='dec', array=np.array(completedec), format='K')
#c3 = fits.Column(name='redshift', array=np.array(completeredshift), format='K')
#c4 = fits.Column(name='redshift_tag', array=np.array(completeredshift_tag), format='5A')
#c5 = fits.Column(name='PMF', array=np.array(complete_epochs), format='K')

#t = fits.BinTableHDU.from_columns([c1, c2, c3])
#t.writeto('table2.fits')
targets.close()
check.close()
