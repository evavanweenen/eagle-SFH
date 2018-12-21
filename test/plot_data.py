from eagle.io import *
from eagle.plot import *
import numpy as np
import matplotlib.pyplot as plt

#EAGLE settings
sim = 'RefL0050N0752' #simulation
cat = 'sdss' #catalogue to build model to
snapshot = 28 #redshift to use

dtype=['<i8','<i8','<f8','<f8','<f8','<f8','<f8','<f8','<f8', '<f8']

xcols = ['sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z']
ycols = ['m_star']

def preprocess(sim, cat, xcols, ycols, dtype, skip_header=15, perc_train = .8):
    #read data
    data = read_data(sim, cat, dtype=dtype, skip_header=skip_header)

    #select redshift
    data = select_redshift(data, snapshot)

    #divide data into x and y
    x, y = divide_input_output(data, xcols, ycols)
    
    #scale data to a logarithmic scale
    x, y = rescale_log(x, y)

    #convert x and y back to a numpy ndarray
    data = data_ndarray(x, y, xcols, ycols)

    return data, x, y

def histogram_individual(data, col, xmax):
    plt.figure()
    plt.hist(data[col], bins=100)
    plt.xlabel(col)
    #plt.xlim((0., xmax))
    plt.savefig('scaled_snap_%s_hist_%s.png'%(snapshot, col))

"""
def sed_mstar(data):
    xnames=['log sdss u', 'log sdss g', 'log sdss r', 'log sdss i', 'log sdss z']; ynames=['log $M_{*}$']
    fig, ax = plt.subplots(1,5, figsize=(20,4), squeeze=True, sharey=True)
    fig.subplots_adjust(wspace=0, hspace=0)
    ax[0].set_ylabel(ynames[0])
    for i, x in enumerate(xnames):  
        ax[i].scatter(data[input_cols[i]], data[output_col], s=.1, color='blue', label='data')
        ax[i].set_xlabel(x)
    plt.legend(title='snapshot=%s'%(snapshot))
    plt.savefig('data_%s_%s_%s_sed-mstar.pdf'%(sim, cat, snapshot))

"""
    
data, x, y = preprocess(sim, cat, xcols, ycols, dtype)

plot_sed_mstar(x, y, 'data', sim, cat, snapshot, predict=False)

"""
for i in input_cols:
    #histogram_individual(data, i, 4e-5)
    #sed_mstar(data, i, 3.6e10, 4e-5)
    histogram_individual(data_scaled, i, 1.)
    sed_mstar(data_scaled[, i, 1., 1.)
    
#histogram_individual(data, output_col, 3.6e10)
histogram_individual(data_scaled, output_col, 1.)
"""
