import numpy as np
import os
import urllib2

df = np.genfromtxt('targets_with3epochs_dr14_spherematch.txt',names=['ra','dec','z','pmf1','pmf2','pmf3'],dtype=(float,float,float,'|S15','|S15','|S15'))
print df

def download_spectra(plate, mjd, fiber, dirname='.'):
    '''  Downloads SDSS spectra from DR14 and puts it in dirname
         Change the SDSS URL to download from a different location
    '''
    FITS_FILENAME = 'spec-%(plate)04i-%(mjd)05i-%(fiber)04i.fits'
    try :
        SDSS_URL = ('https://data.sdss.org/sas/dr14/eboss/spectro/redux/v5_10_0/spectra/%(plate)04i/'
                'spec-%(plate)04i-%(mjd)05i-%(fiber)04i.fits')
        urllib2.urlopen(SDSS_URL % dict(plate=plate,mjd=mjd,fiber=fiber))
        print 'Downloadin from dr14'
    except urllib2.HTTPError as err:
        if err.code == 404 :
            SDSS_URL = ('https://data.sdss.org/sas/dr8/sdss/spectro/redux/26/spectra/%(plate)04i/'
            		'spec-%(plate)04i-%(mjd)05i-%(fiber)04i.fits')
            print 'Downloadin from dr8'
    print SDSS_URL % dict(plate=plate,mjd=mjd,fiber=fiber)
    download_url = 'wget   '+SDSS_URL % dict(plate=plate,mjd=mjd,fiber=fiber)
    print download_url
    os.system(download_url)
    mv_cmd='mv '+FITS_FILENAME % dict(plate=plate,mjd=mjd,fiber=fiber) + ' '+dirname+'/.'
    #print mv_cmd
    os.system(mv_cmd)



def download_spectra_dr14(plate, mjd, fiber, dirname='.'):
    '''  Downloads SDSS spectra from DR14 and puts it in dirname
         Change the SDSS URL to download from a different location
    '''
    FITS_FILENAME = 'spec-%(plate)04i-%(mjd)05i-%(fiber)04i.fits'
    SDSS_URL = ('https://data.sdss.org/sas/dr14/eboss/spectro/redux/v5_10_0/spectra/%(plate)04i/'
            'spec-%(plate)04i-%(mjd)05i-%(fiber)04i.fits')
    #print SDSS_URL % dict(plate=plate,mjd=mjd,fiber=fiber)
    download_url = 'wget -q '+SDSS_URL % dict(plate=plate,mjd=mjd,fiber=fiber)
    print download_url
    os.system(download_url)
    mv_cmd='mv '+FITS_FILENAME % dict(plate=plate,mjd=mjd,fiber=fiber) + ' '+dirname+'/.'
    #print mv_cmd
    os.system(mv_cmd)

for i in range(len(df['pmf1'])):
    print df['pmf1'][i]
    plate = int(df['pmf1'][i].split('-')[0])
    mjd = int(df['pmf1'][i].split('-')[1])
    fiber = int(df['pmf1'][i].split('-')[2])
    print plate,mjd,fiber
    plates = [int(df['pmf1'][i].split('-')[0]),int(df['pmf2'][i].split('-')[0]),int(df['pmf3'][i].split('-')[0])]
    mjds = [int(df['pmf1'][i].split('-')[1]),int(df['pmf2'][i].split('-')[1]),int(df['pmf3'][i].split('-')[1])]
    fibers = [int(df['pmf1'][i].split('-')[2]),int(df['pmf2'][i].split('-')[2]),int(df['pmf3'][i].split('-')[2])]
    for j in range(len(plates)):
        try:
            print j,plates[j],mjds[j],fibers[j]
            download_spectra(plates[j],mjds[j],fibers[j],'Vivek_Sources')
	    print 'tried downloading', plates[j],mjds[j],fibers[j]
	except :
            print 'Need to copy from spAll '
    print '--------------------------------------'
    #ghfjh= raw_input()



write = open('TocopyfromUtah.txt','w')

for i in range(len(df['pmf1'])):
    print 'Working on '+str(i)+'/'+str(len(df))
    plates = [int(df['pmf1'][i].split('-')[0]),int(df['pmf2'][i].split('-')[0]),int(df['pmf3'][i].split('-')[0])]
    mjds = [int(df['pmf1'][i].split('-')[1]),int(df['pmf2'][i].split('-')[1]),int(df['pmf3'][i].split('-')[1])]
    fibers = [int(df['pmf1'][i].split('-')[2]),int(df['pmf2'][i].split('-')[2]),int(df['pmf3'][i].split('-')[2])]
    for j in range(len(plates)):
        if os.path.isfile('spec-{0:04d}-{1:05d}-{2:04d}.fits'.format(plates[j],mjds[j],fibers[j])):
            continue
        else:
            cmd = 'cp /uufs/chpc.utah.edu/common/home/sdss/ebosswork/eboss/spectro/redux/v5_10_7/spectra/full/{0:4d}/spec-{1:04d}-{2:05d}-{3:04d}.fits .'.format(plates[j],plates[j],mjds[j],fibers[j])
            print>>write,cmd
            print cmd

write.close()
