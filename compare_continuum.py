import sys
import os
import numpy as np
import matplotlib.pyplot as plt

alpha1,alpha2,alpha3 = np.loadtxt('ALPHA_AMP_values_V.txt',usecols=(1,2,3)).T
kalpha1,kalpha2,kalpha3 = np.loadtxt('Kate_ALPHA_AMP_values_V.txt',usecols=(1,2,3)).T
print '#Inf in Alpha 1: {}'.format(len(np.isnan(alpha1)==0))
print '#Inf in Alpha 2: {}'.format(len(np.isnan(alpha2)==0))
print '#Inf in Alpha 3: {}'.format(len(np.isnan(alpha3==0)))
print '#Inf in Kate Alpha 1: {}'.format(len(np.isnan(kalpha1)==0))
print '#Inf in Kate Alpha 2: {}'.format(len(np.isnan(kalpha2)==0))
print '#Inf in Kate Alpha 3: {}'.format(len(np.isnan(kalpha3)==0))
print len(np.isnan(alpha2) )
#alpha1 = alpha1[~np.isnan(alpha2)]
#alpha2 = alpha2[~np.isnan(alpha2)]
#alpha3 = alpha3[~np.isnan(alpha2)]
#
xx=np.where(np.isnan(alpha2))
alpha1=np.delete(alpha1,xx)
alpha2=np.delete(alpha2,xx)
alpha3=np.delete(alpha3,xx)
kalpha1=np.delete(kalpha1,xx)
kalpha2=np.delete(kalpha2,xx)
kalpha3=np.delete(kalpha3,xx)

#kalpha1 = alpha1[~np.isnan(alpha2)]
#kalpha2 = alpha2[~np.isnan(alpha2)]
#kalpha3 = alpha3[~np.isnan(alpha2)]

n_bins = 50
fig,(ax0,ax1,ax2) = plt.subplots(1,3,figsize=(15,8))
n1, bins1, patches1 = ax0.hist(kalpha1, n_bins,color='red', normed=0, histtype='step', label='Kate ALPHA')
n, bins, patches = ax0.hist(alpha1, bins1,color='blue', normed=0, histtype='step', label='My ALPHA')

nn1, nbins1, npatches1 = ax1.hist(kalpha2, n_bins,color='red', normed=0, histtype='step', label='Kate ALPHA')
nn, nbins, npatches = ax1.hist(alpha2, nbins1,color='blue', normed=0, histtype='step', label='My ALPHA')

nnn1, nnbins1, nnpatches1 = ax2.hist(kalpha3, n_bins,color='red', normed=0, histtype='step', label='Kate ALPHA')
nnn, nnbins, nnpatches = ax2.hist(alpha3, nnbins1,color='blue', normed=0, histtype='step', label='My ALPHA')

ax0.set_xlabel('ALPHA 1')
ax1.set_xlabel('ALPHA 2')
ax2.set_xlabel('ALPHA 3')
ax0.set_ylabel('Histogram Density')
ax1.set_ylabel('Histogram Density')
ax2.set_ylabel('Histogram Density')
ax0.legend(loc=2)
ax0.set_title('Comparison of Kate\'s and Vivek\'s Continuum Fitting ')
fig.tight_layout()
fig.savefig('Continuumfitting_comparison_V.png')
plt.show()


