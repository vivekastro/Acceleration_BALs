import numpy as np
import os 


df = np.genfromtxt('tdss_allmatches_crop_edit.dat',names=['ra','dec','z','pmf1','pmf2','pmf3'],dtype=(float,float,float,'|S15','|S15','|S15'))

pf = np.genfromtxt('BALcomponents_output.txt',names=('index','filename','vmin','vmax','nt','ntroughs','paddmin','paddmax','ew','ewerr'),dtype=(int,'|S35',float,float,float,float,float,float,float))
print pf['filename']
fpmf = []
fittype =[]
for k in range(len(pf['filename'])):
    fpmf.append(pf['filename'][k].strip().split('_')[1])
    if (pf['filename'][k].strip().split('_')[3].split('.')[0] == 'v'):
        fittype.append('voigt')
    else:
        fittype.append('2guass')
ab=open('linefits.txt','w')
fpmf = np.array(fpmf)
#print fpmf
#print fittype[0]
for i in range(len(df)):
    #print type(df['pmf1'][i]),type(fpmf)
    xx = np.where(fpmf == str(df['pmf1'][i]).strip())[0]
    print xx
    if len(xx)> 0:
        print>>ab,'{}\t{}\t{}\t{}\t{}'.format( i,df['pmf1'][i],df['pmf2'][i],df['pmf3'][i],fittype[xx[0]])

    #dksh=raw_input()
ab.close()
