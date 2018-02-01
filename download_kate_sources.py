import numpy as np
import os

df = np.genfromtxt('tdss_allmatches_crop_edit.dat',names=['ra','dec','z','pmf1','pmf2','pmf3'],dtype=(float,float,float,'|S15','|S15','|S15'))



def download_spectra(plate, mjd, fiber, dirname='.'):
    '''  Downloads SDSS spectra from DR14 and puts it in dirname
         Change the SDSS URL to download from a different location
    '''
    FITS_FILENAME = 'spec-%(plate)04i-%(mjd)05i-%(fiber)04i.fits'
    SDSS_URL = ('https://data.sdss.org/sas/dr14/eboss/spectro/redux/v5_10_0/spectra/%(plate)04i/'
            'spec-%(plate)04i-%(mjd)05i-%(fiber)04i.fits')
    SDSS_URL = ('https://data.sdss.org/sas/dr8/sdss/spectro/redux/26/spectra/%(plate)04i/'
            'spec-%(plate)04i-%(mjd)05i-%(fiber)04i.fits')
    # print SDSS_URL % dict(plate=plate,mjd=mjd,fiber=fiber)
    download_url = 'wget  '+SDSS_URL % dict(plate=plate,mjd=mjd,fiber=fiber)
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
    download_spectra(int(df['pmf1'][i].split('-')[0]),int(df['pmf1'][i].split('-')[1]),int(df['pmf1'][i].split('-')[2]),'Kate_Sources')
    download_spectra(int(df['pmf2'][i].split('-')[0]),int(df['pmf2'][i].split('-')[1]),int(df['pmf2'][i].split('-')[2]),'Kate_Sources')
    download_spectra(int(df['pmf3'][i].split('-')[0]),int(df['pmf3'][i].split('-')[1]),int(df['pmf3'][i].split('-')[2]),'Kate_Sources')

