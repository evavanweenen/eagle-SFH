from eagle.io import *
from eagle.plot import *
from eagle.nn import *

import numpy as np
import tensorflow as tf
import shap
from keras import backend

from argparse import ArgumentParser 
parser = ArgumentParser()
parser.add_argument("input", help = "Choose nocolors, subsetcolors or allcolors")
parser.add_argument("sampling", help = "Choose None, random or uniform")
args = parser.parse_args()

if args.sampling == 'None': args.sampling = None
assert args.input == 'nocolors' or args.input == 'subsetcolors' or args.input == 'allcolors'
assert args.sampling == None or args.sampling == 'random' or args.sampling == 'uniform'

seed = 7
np.random.seed(seed) #fix seed for reproducibility
tf.set_random_seed(seed)

plotting = True

inp = args.input #'nocolors'#'subsetcolors'#'allcolors'#

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

#input output plot
xnames=[dust + ' ' + cat + ' ' + i for i in list(fluxes) + [c[0] + '-' + c[1] for c in colors]]
ynames=['$\log_{10} M_{*} (M_{\odot})$']#['$\log_{10}$ SFR $(M_{\odot} yr^{-1})$']#

#preprocessing
cs = False ; cs_mask = None #central satellite analysis (only change cs)
sampling = 'uniform' ; bins = 10 ; count = 125 ; N = 625 ; scaling = False if sampling == 'uniform' else True #both same uniform mass distribution

"""
#ML settings GridSearch
h_nodes = [20, 50, 50]
activation = ['tanh', 'linear', 'tanh', 'linear']
optimizer = 'Adamax'
"""
if inp == 'nocolors':
    #Hyperas no colors
    h_nodes = [40, 50, 30]
    activation = ['linear', 'tanh', 'tanh', 'linear']
elif inp == 'subsetcolors':
    #Hyperas subset colors
    h_nodes = [50, 50, 20]
    activation = ['relu', 'tanh', 'linear', 'linear']
elif inp == 'allcolors':
    #Hyperas all colors
    h_nodes = [30, 30, 50]
    activation = ['linear', 'tanh', 'relu', 'linear']

optimizer = 'Adam'
dropout = [0., 0., 0.]
perc_train = .8 #todo: this is not used anywhere, give it to io
loss = 'mean_absolute_error'
epochs = 15
batch_size = 128

def cross_validation(X, Y, K, seed):
    kf = KFold(n_splits=K, random_state=seed)
    X_train = [] ; Y_train = []; X_val = []; Y_val = []
    for idx_train, idx_val in kf.split(X):
        X_train.append(X[idx_train])
        Y_train.append(Y[idx_train])
        X_val.append(X[idx_val])
        Y_val.append(Y[idx_val])       
    return X_train, Y_train, X_val, Y_val

def add_scores(score, x, y, y_pred):
    y.reshape(len(y),) ; y_pred.reshape(len(y_pred),)
    score[1] = r2_score(y, y_pred)
    return np.append(score, [adjusted_R_squared(score[1], x.shape[1], len(y)), np.mean(y_pred - y), np.var(y_pred - y)])

#------------------------------------------ Read data --------------------------------------------------------------
#read data
eagle = EAGLE(fol, sim, cat, dust, snap, redshift, seed)
eagle.preprocess(colors, eagle_xcols, eagle_ycols, eagle_xtype, sampling, bins, count, N, scaling=scaling)

#sdss
sdss = SDSS(sdss_xcols, sdss_ycols, sdss_xtype, redshift)
sdss.preprocess(eagle, colors, sampling, scaling=scaling)

if sampling == 'uniform':
    uniform_mass_sampling(sdss, ref_pre=eagle, count=count, cs_arr=False)
    uniform_mass_sampling(eagle, ref_post=sdss, count=int(count*0.2), cs_arr=True)
    random_sampling(eagle, int(N*0.2))
    random_sampling(sdss, N, cs_arr=False) #TODO
    eagle.scaling()
    sdss.scaling(eagle)

x_train, x_test, y_train, y_test = train_test_split(sdss.x, sdss.y, test_size=1-perc_train, random_state=seed, shuffle=True)
print("EAGLE: len(eagle.x) = %s ; len(x_train) = %s ; len(x_test) = %s ; len(sdss.x) = %s"%(len(eagle.x), len(x_train), len(x_test), len(sdss.x)))

if plotting:
    if sampling == 'uniform':
        edges = eagle.yscaler.transform(eagle.edges.reshape(-1,1)).T[0]
    else:
        edges = 7
    plot_data = PLOT_DATA(xnames+ynames, sim=sim, cat=cat, snap=snap, N=len(sdss.y), inp=inp, sampling=str(sampling))
    plot_data.hist_data(('sdss-train', 'sdss-test', 'eagle-total'), [np.hstack((x_train, y_train)), np.hstack((x_test, y_test)), np.hstack((eagle.x, eagle.y))], eagle.yscaler.transform(eagle.edges.reshape(-1,1)).T[0], xlim=[-1.1,1.1], ylim=[-1.1,1.1])

input_size = len(x_train[0])
output_size = len(y_train[0])
nodes = [input_size] + h_nodes + [output_size]
print("nodes in the network: ", nodes)

#read hyperparameters 
nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, epochs, batch_size, optimizer)

#------------------------------------------Cross-validate on SDSS--------------------------------------------------------------
K = 5

#crossvalidation data
X_train, Y_train, X_val, Y_val = cross_validation(x_train, y_train, K, seed)

cv_score = np.empty((K, 5))
for i in range(K):
      #make architecture of the network
      nn.MLP_model()
      result = nn.model.fit(X_train[i], Y_train[i], batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_val[i], Y_val[i]))
      cv_score[i,:2] = nn.model.evaluate(X_val[i], Y_val[i], verbose=0)
      y_pred = nn.model.predict(X_val[i]) #predict
      cv_score[i] = add_scores(cv_score[i,:2], X_val[i], Y_val[i], y_pred) #add scores
avg_cv_score = np.average(cv_score, axis=0)

print("Cross-validated errors (SDSS): MAE = %.3e ; R^2 = %.3f ; adjusted R^2 = %.3f ; error_mean = %.3e ; error_sigma = %.3e"%tuple(avg_cv_score))

#------------------------------------------Train and test on SDSS--------------------------------------------------------------
#train, evaluate, predict, postprocess
nn.MLP_model()
result = nn.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test)) #train
score = nn.model.evaluate(x_test, y_test, verbose=0) #evaluate
y_pred = nn.model.predict(x_test) #predict
score = add_scores(score, x_test, y_test, y_pred) #add scores
print("Errors (SDSS): MAE = %.3e ; R^2 = %.3f ; adjusted R^2 = %.3f ; error_mean = %.3e ; error_sigma = %.3e"%tuple(score))

#------------------------------------------Compare with EAGLE--------------------------------------------------------------
#evaluate, predict, postprocess
eagle_score = nn.model.evaluate(eagle.x, eagle.y, verbose=0) #evaluate
eagle.ypred = nn.model.predict(eagle.x) #predict
eagle_score = add_scores(eagle_score, eagle.x, eagle.y, eagle.ypred)
print("Errors (EAGLE): MAE = %.3e ; R^2 = %.3g ; adjusted R^2 = %.3g ; error_mean = %.3e ; error_sigma = %.3e"%tuple(eagle_score))

np.save('/disks/strw9/vanweenen/mrp2/plots/score_inverse_'+eagle_ycols[0]+'_inp='+inp+'_sampling='+str(sampling)+'.pdf', np.vstack((avg_cv_score, score, eagle_score)))

#------------------------------------------Plotting--------------------------------------------------------------
x_test, y_test, y_pred = sdss.postprocess(eagle, x_test, y_test, y_pred) #postprocessing

#plot
if plotting:
      plot = PLOT_NN(eagle, nn, xnames, ynames, dataname='sdss', N=len(y_test), inp=inp, sampling=str(sampling), score=score)#[mse[19], r2[19]]
      plot.plot_learning_curve(result)
      plot.plot_input_output(x_test, y_test, y_pred, 'scatter') #scatter, contour
      plot.plot_true_predict(y_test, y_pred, 'scatter', cs_mask) #scatter, hexbin, contour, contourf
      plot.plot_output_error(y_test, y_pred, 'contourf', cs_mask) #scatter, hexbin, contour

eagle.x, eagle.y, eagle.ypred = eagle.postprocess(eagle.x, eagle.y, eagle.ypred) #postprocessing

#plot
if plotting:
      plot = PLOT_NN(eagle, nn, xnames, ynames, dataname='eagle', N=len(eagle.y), inp=inp, sampling=str(sampling), score=eagle_score)#[mse[19], r2[19]]
      plot.plot_input_output(eagle.x, eagle.y, eagle.ypred) #scatter, contour
      plot.plot_true_predict(eagle.y, eagle.ypred, 'scatter') #scatter, hexbin, contour, contourf
      plot.plot_output_error(eagle.y, eagle.ypred, 'contourf', bins=10, ylim=(-0.4, 0.05)) #scatter, hexbin, contour

