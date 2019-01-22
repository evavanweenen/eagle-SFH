from eagle.io import *
from eagle.plot import *

import numpy as np
from copy import copy

#EAGLE settings
fol = ''
sim = 'RefL0100N1504' #simulation
cat = 'dusty-sdss' #catalogue to build model to
snap = 27 #redshift to use

#------------------------------EAGLE---------------------------------------------------
eagle_xcols = ['dusty_sdss_' + f for f in ['u','g','r','i','z']]
eagle_ycols = ['m_star']
eagle_datacols = eagle_xcols + eagle_ycols

eagle = EAGLE(fol, sim, cat, snap, eagle_xcols, eagle_ycols, perc_train=.8)
eagle.read_data()
select_cols(eagle)

eagle.data_log = rescale_log(np.array(eagle.data.tolist()))
#x_train, y_train, x_test, y_test = eagle.preprocess()


#------------------------------SDSS----------------------------------------------------
sdss_xcols = ['flux_u', 'flux_g', 'flux_r', 'flux_i', 'flux_z']
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


#------------------------------PLOT----------------------------------------------------
#input output
xnames=['$\log_{10}$ dusty sdss flux ' + f + ' (Jy) ' for f in ['u','g','r','i','z']]
ynames=['$\log_{10} M_{*} (M_{\odot})$']#['log SFR']
datanames = xnames + ynames

plot = PLOT_DATA(datanames, sim=sim, cat=cat, snap=snap)
plot.hist_data(('eagle', 'sdss'), [eagle.data_log, sdss.data_log])
