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


for i in  range(len(pf)):
    wave,flux,err = np.loadtxt(pf['filename'][i]).T
    vel = velocity(wave1, cw, 0.0)
    xx = np.where((vel >= pf['vmin'][i] - pf['paddmin'][i]) & ( vel <= pf['vmax'][i]+pf['paddmax'][i]))[]
    nvel=vel[xx]/1000 ; nflux = flux[xx] ; nerr = 1.0/np.sqrt(err[xx])
    sfilename = 'spec-'+pf['filename'][i].split(_)[1]+pf['filename'][i].split(_)[2]pf['filename'][i].split(_)[3]
    savefilename = 'fivecol/'+sfilename+'.crop.5col_0'
    print savefilename
    #np.savetxt(savefilename,zip(nvel,nflux,nerr,nflux,nerr))
