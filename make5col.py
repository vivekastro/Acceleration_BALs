import numpy as np
import os


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

cw=1549.0

pf = np.genfromtxt('BALcomponents_output.txt',names=('index','filename','vmin','vmax','nt','ntroughs','paddmin','paddmax','ew','ewerr'),dtype=(int,'|S35',float,float,float,float,float,float,float))
uindex = np.unique(pf['index'])
for i in  range(len(uindex)):
    match = np.where(pf['index'] == uindex[i])[0]
    mpf = pf[match]
    for k in range(len(mpf)):
        wave,flux,err = np.loadtxt('EmissionLines_Norm_Spectra/'+str(mpf['filename'][k])).T
        vel = velocity(wave, cw, 0.0)/1000.
        print 'Match', match, uindex[i]
        print mpf
        ntroughs = mpf['ntroughs'][0]
        print 'Ntroughs: ',ntroughs
        for ii in range(int(ntroughs)):
            print uindex[i]
            print 'index',ii,(mpf['vmax'][ii] - mpf['paddmin'][ii])/1000, (mpf['vmin'][ii] + mpf['paddmax'][ii])/1000.
            xx = np.where((vel >= (mpf['vmax'][ii] - mpf['paddmin'][ii])/1000.) & ( vel <= (mpf['vmin'][ii]+mpf['paddmax'][ii])/1000.))[0]
            nvel=vel[xx] ; nflux = flux[xx] ; nerr = err[xx]
            print 'Final',min(nvel),max(nvel)
            sfilename = 'spec-'+mpf['filename'][k].split('_')[1]+'_'+mpf['filename'][k].split('_')[2]+'_'+mpf['filename'][k].split('_')[3]
            savefilename = 'fivecol/'+sfilename+'.crop.5col'+'_'+str(ii)
            print savefilename
            np.savetxt(savefilename,zip(nvel,nflux,nerr,nflux,nerr), fmt='%10.5f')
            #fsdf=raw_input()
