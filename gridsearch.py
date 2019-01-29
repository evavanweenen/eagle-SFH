from eagle.io import *
from eagle.plot import *
from eagle.nn import *

import numpy as np
from scipy.sparse import diags

import datetime
import sys

seed = 7
np.random.seed(seed) #fix seed for reproducibility

#EAGLE settings
fol = ''
sim = 'RefL0100N1504' #simulation
cat = 'dusty-sdss' #catalogue to build model to
snap = 27 #snapshot to use
redshift = 0.1

#input output
inp = 'dusty-sdss'
outp = 'mstar'#'sfr'
xnames=['$\log_{10}$ dusty sdss flux ' + f + ' (Jy) ' for f in ['u','g','r','i','z']]
ynames=['$\log_{10} M_{*} (M_{\odot})$']#['log SFR']

#eagle
eagle_xtype = 'flux'
eagle_xcols = ['dusty' + eagle_xtype + '_sdss_' + f for f in ['u','g','r','i','z']]
eagle_ycols = ['m_star']#['sfr']

#sdss
sdss_xcols = ['flux_' + f for f in ['u', 'g', 'r', 'i', 'z']]
sdss_ycols = ['Mstellar_median']
sdss_datacols = sdss_xcols + sdss_ycols + ['Redshift']

#------------------------------------------GridSearch--------------------------------------------------------------
#write output terminal to file
path =  '/disks/strw9/vanweenen/mrp2/optimization/'
name = ''

date = datetime.date.today().strftime('%Y%m%d')
f = open(path+'gridsearch_'+date+name+'.out', 'w')
sys.stdout = f

#hyperparams
perc_train = .8
h_nodes = [[20,40]]+2*[[None, 20, 40]]#h_nodes = 3*[[5, 10, 20, 30, 40, 50, 60]]
dropout = [0., 0., 0.]#3*[[0., .4, .8]]#dropout = 3*[[0, 0.2, 0.4, 0.6, 0.8, 1.]]
activation = 3*[['sigmoid', 'tanh', 'relu']]+['linear']#'relu', 'sigmoid', 'linear'
loss = 'mean_squared_error'
lr_rate = 0.001
epochs = 20
batch_size = 64
optimizer = 'Adam'#['sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

#read data
eagle = EAGLE(fol, sim, cat, snap, redshift, eagle_xcols, eagle_ycols, perc_train)
x_train, y_train, x_test, y_test = eagle.preprocess(eagle_xtype)

input_size = len(x_train[0])
output_size = len(y_train[0])
nodes = [input_size] + h_nodes + [output_size]
print("nodes in the network: ", nodes)

#read hyperparameters and make architecture of the network
nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, lr_rate, epochs, batch_size, optimizer)
nn.GridSearch(x_train, y_train, 2)

#save output
grid_labels = ['h_nodes0', 'activation0', 'h_nodes1', 'activation1', 'h_nodes2', 'activation2']
grid_names = ['h nodes0', 'activation0', 'h nodes1', 'activation1', 'h nodes2', 'activation2']
grid_edges = [nn.h_nodes0, nn.activation0, nn.h_nodes1, nn.activation1, nn.h_nodes2, nn.activation2]

grid_params = [nn.grid_result.cv_results_['param_'+i].filled() for i in grid_labels]

grid_shape = [len(i) for i in grid_edges]

grid_values = nn.means.reshape(grid_shape)

np.save(path + 'mse.npy', grid_values)

#plotting
dim = len(grid_labels)

fig, ax = plt.subplots(dim, dim, figsize=(7,6), squeeze=True)#, sharex=True, sharey=True)
vmin = np.amin(grid_values) ; vmax = np.amax(grid_values)
for i in range(dim):
    for j in range(dim):
        axis = tuple([x for x in np.arange(dim) if x!=i and x!=j])
        if i == j:
            diag = np.diagflat(np.average(grid_values, axis=axis))
            diag[diag == 0] = np.nan
            im = ax[i,j].imshow(diag, vmin=vmin, vmax=vmax, cmap='viridis', aspect='auto')
        if i < j:
            im = ax[i,j].imshow(np.average(grid_values, axis=axis), vmin=vmin, vmax=vmax, cmap='viridis', aspect='auto')
        elif j < i:
            im = ax[i,j].imshow(np.average(grid_values, axis=axis).T, vmin=vmin, vmax=vmax, cmap='viridis', aspect='auto')        
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])        
        ax[dim-1,j].set_xticks(np.arange(len(grid_edges[j])))
        ax[dim-1,j].set_xticklabels(np.array(grid_edges[j]).astype('str').tolist())
        ax[dim-1,j].set_xlabel(grid_names[j])
    ax[i,0].set_ylabel(grid_names[i])
    ax[i,0].set_yticks(np.arange(len(grid_edges[i])))  
    ax[i,0].set_yticklabels(np.array(grid_edges[i]).astype('str').tolist())    
cbar = fig.colorbar(im, ax=ax)
cbar.ax.set_title('average MSE')
plt.savefig(path+'GridSearch.pdf')
plt.show()
