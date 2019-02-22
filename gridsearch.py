from eagle.io import *
from eagle.plot import *
from eagle.nn import *

import numpy as np
import scipy as sp
from scipy.sparse import diags

import datetime
import sys
import tensorflow as tf

seed = 7
np.random.seed(seed) #fix seed for reproducibility
tf.set_random_seed(seed)

path =  '/disks/strw9/vanweenen/mrp2/optimization/'
name = ''
"""
#write output terminal to file
name = ''
date = datetime.date.today().strftime('%Y%m%d')
f = open(path+'gridsearch_'+date+name+'.out', 'w')
sys.stdout = f
"""

#EAGLE settings
fol = ''
sim = 'RefL0100N1504'   # simulation
cat = 'sdss'            # catalogue to build model to
dust = 'dusty'          # '' if no dust
snap = 27               # snapshot to use
redshift = 0.1          # redshift of snapshot    

fluxes = ('u', 'g', 'r', 'i', 'z')
colors = ()#('ug', 'gr', 'ri', 'iz')
xcols = fluxes + colors

#eagle
eagle_xtype = 'flux'
eagle_xcols = [dust + eagle_xtype + '_' + cat + '_' + f for f in xcols]
eagle_ycols = ['m_star']#['sfr']#

#sdss
sdss_xtype = 'flux'
sdss_xcols = [sdss_xtype + '_' + f for f in xcols]
sdss_ycols = ['Mstellar_median']#['SFR_median']#

#input output plot
xnames=[dust + ' ' + cat + ' ' + i for i in list(fluxes) + [c[0] + '-' + c[1] for c in colors]]
ynames=['$\log_{10} M_{*} (M_{\odot})$']#['$\log_{10}$ SFR $(M_{\odot} yr^{-1})$']#

equal_bins = True #uniform mass distribution


#------------------------------------------GridSearch--------------------------------------------------------------
#hyperparams
perc_train = .8
h_nodes = [20, 50, 50]#[[10, 20, 30, 40, 50]]+2*[[None, 10, 20, 30, 40, 50]]#
dropout = [0, 0, 0]#3*[[0., 0.2, 0.4, 0.7]]#
activation = ['tanh', 'linear', 'tanh', 'linear']#3*[['sigmoid', 'tanh', 'relu', 'linear']] + ['linear']#
loss = 'mean_squared_error'
lr_rate = 0.001
epochs = 15
batch_size = 128
optimizer = ['sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']#'Adam'#

#read data
eagle = EAGLE(fol, sim, cat, dust, snap, redshift)
eagle.read_data()
x_train, y_train, x_test, y_test = eagle.preprocess(colors, eagle_xcols, eagle_ycols, eagle_xtype, equal_bins)

input_size = len(x_train[0])
output_size = len(y_train[0])
nodes = [input_size] + h_nodes + [output_size]
print("nodes in the network: ", nodes)

#read hyperparameters and make architecture of the network
nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, lr_rate, epochs, batch_size, optimizer)
nn.GridSearch(x_train, y_train, 3)

#save output
grid_edges = optimizer #param grid: list with lists of options
grid_names = ['optimizer'] #make sure this name is latex compatible
grid_shape = [len(i) for i in grid_edges]

grid_mse = nn.grid_result.cv_results_['mean_test_neg_mean_squared_error'].reshape(grid_shape) * -1
grid_r2 = nn.grid_result.cv_results_['mean_test_r2'].reshape(grid_shape)

np.save(path + 'mse' + str(grid_names) + name + '.npy', grid_mse)
np.save(path + 'r2' + str(grid_names) + name + '.npy', grid_r2)

"""
grid_mse = np.load(path+'mse'+str(grid_names)+ name + '.npy')
grid_r2 = np.load(path+'r2'+str(grid_names)+ name + '.npy')
"""

print("MSE: Best models")
idxs_mse = np.argsort(grid_mse, axis=None)[:3]
for m in range(3):
    idx_mse = np.unravel_index(idxs_mse[m], grid_mse.shape)
    print("number ", m,  'with mse: %.4e r2: %.4f'%(grid_mse[idx_mse], grid_r2[idx_mse]))
    for i, idx in enumerate(idx_mse):
        print(grid_names[i], grid_edges[i][idx])
    

print("r2: Best models")
idxs_r2 = np.argsort(grid_r2, axis=None)[::-1][:3]
for m in range(3):
    idx_r2 = np.unravel_index(idxs_r2[m], grid_r2.shape)
    print("number ", m,  'with mse: %.4e r2: %.4f'%(grid_mse[idx_r2], grid_r2[idx_r2]))
    for i, idx in enumerate(idx_r2):
        print(grid_names[i], grid_edges[i][idx])

#plotting
plot_gridsearch('mse', grid_mse, grid_edges, grid_names, name)
plot_gridsearch('r2', grid_r2, grid_edges, grid_names, name)


"""
#------------------------------------------RandomizedSearch--------------------------------------------------------------
#hyperparams
perc_train = .8
h_nodes = [[10, 20, 30, 40, 50]] + 2*[[None, 10, 20, 30, 40, 50]]
dropout = 3*[sp.stats.uniform(0., 1.)]
activation = 3*[['sigmoid', 'tanh', 'relu']]+['linear']#'relu', 'sigmoid', 'linear'
loss = 'mean_squared_error'
lr_rate = 0.001
epochs = 20
batch_size = 64
optimizer = ['sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

#read data
eagle = EAGLE(fol, sim, cat, snap, redshift, eagle_xcols, eagle_ycols, perc_train)
x_train, y_train, x_test, y_test = eagle.preprocess(eagle_xtype)

input_size = len(x_train[0])
output_size = len(y_train[0])
nodes = [input_size] + h_nodes + [output_size]
print("nodes in the network: ", nodes)

#read hyperparameters and make architecture of the network
nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, lr_rate, epochs, batch_size, optimizer)
nn.GridSearch(x_train, y_train, 2, n_iter=10)

"""




