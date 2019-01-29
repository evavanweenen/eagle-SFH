from eagle.io import *
from eagle.plot import *
from eagle.calc import *

import numpy as np
from copy import copy

#EAGLE settings
fol = ''
sim = 'RefL0100N1504' #simulation
cat = 'dusty-sdss' #catalogue to build model to
snap = 27 #redshift to use
redshift = 0.1

def select_mass_range(data, minmass, maxmass):
    data = data[data[:,5] > minmass]
    data = data[data[:,5] < maxmass]
    return data

def EAGLE_data(eagle_xtype, scaling=False):
    #eagle_xtype = 'magnitude' or 'flux'
    eagle_xcols = ['dusty' + eagle_xtype + '_sdss_' + f for f in ['u','g','r','i','z']]
    eagle_ycols = ['m_star']
    eagle_datacols = eagle_xcols + eagle_ycols
    
    eagle = EAGLE(fol, sim, cat, snap, redshift, eagle_xcols, eagle_ycols, perc_train=.8)
    eagle.preprocess(eagle_xtype, scaling=scaling)
    
    return eagle, merge_x_y(eagle.x, eagle.y)

def SDSS_data(sdss_xtype, eagle, frac = 5e-3, scaling=False):
    #sdss_xtype = '_bc' or ''
    sdss_xcols = ['flux_' + f + sdss_xtype for f in ['u', 'g', 'r', 'i', 'z']]
    sdss_ycols = ['Mstellar_median']
    sdss_datacols = sdss_xcols + sdss_ycols

    #do all preprocessing steps except for scaling
    sdss = SDSS(sdss_datacols, sdss_xcols, sdss_ycols, redshift = 0.1)
    sdss.preprocess(eagle, frac=frac, scaling=scaling)
    
    return merge_x_y(sdss.x, sdss.y)

#------------------------------EAGLE---------------------------------------------------
eagle_df, eagle_dustyflux = EAGLE_data('flux', scaling=False) #dusty fluxes
eagle_dm, eagle_dustymag = EAGLE_data('magnitude', scaling=False) #dusty magnitudes

#eagle_dustyflux = select_mass_range(eagle_dustyflux, 9.5, 10.5)
#eagle_dustymag = select_mass_range(eagle_dustymag, 9.5, 10.5)

#------------------------------SDSS----------------------------------------------------
sdss_ac = SDSS_data('', eagle_df, frac=5e-3, scaling=False) #flux after calibration
sdss_bc = SDSS_data('_bc', eagle_df, frac=5e-3, scaling=False) #flux before calibration

#sdss_ac = select_mass_range(sdss_ac, 9.5, 10.5)
#sdss_bc = select_mass_range(sdss_bc, 9.5, 10.5)

#------------------------------PLOT----------------------------------------------------
#input output
xnames=['$\log_{10}$ dusty sdss flux ' + f + ' (Jy) ' for f in ['u','g','r','i','z']]
ynames=['$\log_{10} M_{*} (M_{\odot})$']#['log SFR']
datanames = xnames + ynames

plot = PLOT_DATA(datanames, sim=sim, cat=cat, snap=snap)
plot.hist_data(('eagle-dustyflux', 'eagle-dustymag', 'sdss-beforecalibration', 'sdss-aftercalibration'), [eagle_dustyflux, eagle_dustymag, sdss_ac, sdss_bc])
