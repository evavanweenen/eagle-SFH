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

def EAGLE_data(eagle_xtype):
    #eagle_xtype = 'magnitude'
    eagle_xcols = ['dusty' + eagle_xtype + '_sdss_' + f for f in ['u','g','r','i','z']]
    eagle_ycols = ['m_star']
    eagle_datacols = eagle_xcols + eagle_ycols

    eagle = EAGLE(fol, sim, cat, snap, redshift, eagle_xcols, eagle_ycols, perc_train=.8)
    eagle.read_data()
    select_cols(eagle)

    eagle.data_arr = np.array(eagle.data.tolist())
    if eagle_xtype == 'magnitude':
        d_L = luminosity_distance(redshift)
        eagle.data_arr[:,:5] = magAB_to_flux(abs_to_app_mag(eagle.data_arr[:,:5], d_L))

    eagle.data_log = rescale_log(eagle.data_arr)
    #x_train, y_train, x_test, y_test = eagle.preprocess()
    return eagle.data_log


def SDSS_data(sdss_xtype):
    #sdss_xtype = '_bc' or ''
    sdss_xcols = ['flux_' + f + sdss_xtype for f in ['u', 'g', 'r', 'i', 'z']]
    sdss_ycols = ['Mstellar_median']
    sdss_datacols = sdss_xcols + sdss_ycols

    #do all preprocessing steps except for scaling
    sdss = SDSS(sdss_datacols+['Redshift'], sdss_xcols, sdss_ycols, redshift = 0.1)
    sdss.read_data()
    sdss.select_redshift()
    sdss.datacols = sdss_datacols ; select_cols(sdss) #remove redshift column

    sdss.data_log = np.array(sdss.data.tolist())
    sdss.data_log[:,:5] = rescale_log(sdss.data_log[:,:5]) #logscaling

    data_mask = ~np.isinf(sdss.data_log).any(axis=1)
    sdss.data_log = sdss.data_log[data_mask] #remove infinite value rows
    #x_sdss, y_sdss = sdss.preprocess()
    return sdss.data_log

#------------------------------EAGLE---------------------------------------------------
eagle_dustyflux = EAGLE_data('flux')
eagle_dustymag = EAGLE_data('magnitude') #dustymagnitude is the source of the data, eagle_dustymag contains fluxes

#------------------------------SDSS----------------------------------------------------
sdss_ac = SDSS_data('') #flux after calibration
sdss_bc = SDSS_data('_bc') #flux before calibration

#------------------------------PLOT----------------------------------------------------
#input output
xnames=['$\log_{10}$ dusty sdss flux ' + f + ' (Jy) ' for f in ['u','g','r','i','z']]
ynames=['$\log_{10} M_{*} (M_{\odot})$']#['log SFR']
datanames = xnames + ynames

plot = PLOT_DATA(datanames, sim=sim, cat=cat, snap=snap)
#plot.hist_data(('eagle', 'sdss'), [eagle.data_log, sdss.data_log])
plot.hist_data(('eagle-dustyflux', 'eagle-dustymag', 'sdss-beforecalibration', 'sdss-aftercalibration'), [eagle_dustyflux, eagle_dustymag, sdss_ac, sdss_bc])
