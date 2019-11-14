from eagle.io import *
from eagle.plot import *
from eagle.nn import *

import numpy as np
import tensorflow as tf
import shap
from keras import backend

inp = 'allcolors'

sampling = 'none'

seed = 7
np.random.seed(seed) #fix seed for reproducibility
tf.set_random_seed(seed)


plotting = True

#EAGLE settings
fol = ''
sim = 'RefL0100N1504'   # simulation
cat = 'sdss'            # catalogue to build model to
dust = 'dusty'          # '' if no dust
snap = 27               # snapshot to use
redshift = 1.0063854e-01#0.1# redshift of snapshot

fluxes =  ('u', 'g', 'r', 'i', 'z')
if inp == 'nocolors':
    colors = ()
elif inp == 'subsetcolors':
    colors = ('ug', 'gr', 'ri', 'iz')
elif inp == 'allcolors':
    colors = ('ug', 'ur', 'ui', 'uz', 'gr', 'gi', 'gz', 'ri', 'rz', 'iz')


xcols = fluxes + colors
xcols_string = ','.join(xcols)

#eagle
eagle_xtype = 'flux'
eagle_xcols = [dust + eagle_xtype + '_' + cat + '_' + f for f in xcols]
eagle_ycols = ['m_star']#['sfr']#

#sdss
sdss_xtype = 'flux'
sdss_xcols = [sdss_xtype + '_' + f for f in xcols]
sdss_ycols = ['Mstellar_median']#['SFR_median']#

xnames= list(fluxes) + [c[0] + '-' + c[1] for c in colors]
ynames=['$\log_{10} M_{*} (M_{\odot})$']#['$\log_{10}$ SFR $(M_{\odot} yr^{-1})$']#

N=125

arr = [0, 1e10, 2e10, 3e10, 4e10, 5e10, 6e10, 7e10, 8e10, 9e10, 10e10, 11e10]

for i in range(len(arr)-1):
    #------------------------------------------ Read data --------------------------------------------------------------
    #read data
    eagle = EAGLE(fol, sim, cat, dust, snap, redshift, seed, eagle_xcols, eagle_ycols, eagle_xtype)
    eagle.preprocess(colors)
    
    sdss = SDSS(sdss_xcols, sdss_ycols, sdss_xtype, redshift)
    sdss.preprocess(colors)
    
    eagle.x = eagle.x[10**eagle.y > arr[i]]
    eagle.y = eagle.y[10**eagle.y > arr[i]]
    eagle.x = eagle.x[10**eagle.y < arr[i+1]]
    eagle.y = eagle.y[10**eagle.y < arr[i+1]]
    
    sdss.x = sdss.x[10**sdss.y > arr[i]]
    sdss.y = sdss.y[10**sdss.y > arr[i]]
    sdss.x = sdss.x[10**sdss.y < arr[i+1]]
    sdss.y = sdss.y[10**sdss.y < arr[i+1]]
    
    random_sampling(eagle, N)
    random_sampling(sdss, N, False)
    
    eagle.scaling()
    sdss.scaling(eagle)
    
    plot_data =  PLOT_DATA(xnames+ynames, sim=sim, snap=snap, N=len(sdss.y), inp=inp, sampling=str(sampling))
    edges = 10
    
    plot_data.hist_data(('eagle%.1e-%.1e'%(arr[i],arr[i+1]), 'sdss%.1e-%.1e'%(arr[i], arr[i+1])), [np.hstack((eagle.x, eagle.y)), np.hstack((sdss.x, sdss.y))], edges, xlim=[-1.5,1.5], ylim=[-1.1,1.1])
