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
#redshift_array = np.flip(np.array([2.2204460e-16, 1.0063854e-01, 1.8270987e-01, 2.7090108e-01, 3.6566857e-01, 5.0310730e-01, 6.1518980e-01, 7.3562960e-01, 8.6505055e-01, 1.0041217e+00, 1.2593315e+00, 1.4867073e+00, 1.7369658e+00, 2.0124102e+00, 2.2370370e+00, 2.4784133e+00, 3.0165045e+00, 3.5279765e+00, 3.9836636e+00, 4.4852138e+00, 5.0372367e+00, 5.4874153e+00, 5.9711623e+00, 7.0495663e+00, 8.0746160e+00, 8.9878750e+00, 9.9930330e+00]))
#redshift = redshift_array[snap - 2]

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
#xnames=[dust + ' ' + cat + ' ' + i for i in list(fluxes) + [c[0] + '-' + c[1] for c in colors]]
xnames= list(fluxes) + [c[0] + '-' + c[1] for c in colors]
ynames=['$\log_{10} M_{*} (M_{\odot})$']#['$\log_{10}$ SFR $(M_{\odot} yr^{-1})$']#

#preprocessing
cs = False ; cs_mask = None #central satellite analysis (only change cs)
sampling = args.sampling #'uniform'
superuniform = False

#both same uniform mass distribution
if superuniform:
    bins = 300 ; count = 5 ; N = 625 #both same uniform mass distribution
else: 
    bins = 10 ; count = 125 ; N = 625 

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

"""
def scaling_sdss(self):
    if self.x.ndim == 1:
        self.x = self.x.reshape(-1,1)
    self.y = self.y.reshape(-1,1)
    self.xscaler = MinMaxScaler(feature_range=(-1,1)).fit(self.x)#StandardScaler())#
    self.yscaler = MinMaxScaler(feature_range=(-1,1)).fit(self.y)#StandardScaler())#
    self.x = self.xscaler.transform(self.x)
    self.y = self.yscaler.transform(self.y)

def scaling_eagle(self, sdss):
    self.y = self.y.reshape(-1,1)
    if self.x.ndim == 1:
        self.x = self.x.reshape(-1,1)
    self.x = sdss.xscaler.transform(self.x)
    self.y = sdss.yscaler.transform(self.y)
    
def postprocess_sdss(self, x_test, y_test, y_pred):
    #return x_test, y_test and y_pred to their original scale  
    x_test = self.xscaler.inverse_transform(x_test)
    y_test = self.yscaler.inverse_transform(y_test)
    y_pred = self.yscaler.inverse_transform(y_pred)
    return x_test, y_test, y_pred

def postprocess_eagle(self, sdss, x, y, y_pred):
    x = sdss.xscaler.inverse_transform(x)
    y = sdss.yscaler.inverse_transform(y)
    y_pred = sdss.yscaler.inverse_transform(y_pred)
    return x, y, y_pred
"""

#------------------------------------------ Read data --------------------------------------------------------------
#read data
eagle = EAGLE(fol, sim, cat, dust, snap, redshift, seed, eagle_xcols, eagle_ycols, eagle_xtype)
eagle.preprocess(colors)

sdss = SDSS(sdss_xcols, sdss_ycols, sdss_xtype, redshift)
sdss.preprocess(colors)

#sample data
sample(sdss, eagle, sampling, bins, count, N)

#scale data
eagle.scaling()
sdss.scaling(eagle)

#divide into train and test set
x_train, x_test, y_train, y_test = train_test_split(sdss.x, sdss.y, test_size=1-perc_train, random_state=seed, shuffle=True)
print("Total size of data: %s; size of training set: %s ; size of test set: %s"%(len(eagle.x), len(x_train), len(x_test)))

print("EAGLE: len(eagle.x) = %s ; len(x_train) = %s ; len(x_test) = %s ; len(sdss.x) = %s"%(len(eagle.x), len(x_train), len(x_test), len(sdss.x)))

if plotting:
    #if sampling == 'uniform':
        #edges = eagle.yscaler.transform(eagle.edges.reshape(-1,1)).T[0]
    #else:
        #edges = 7
    edges = 10
    plot_data = PLOT_DATA(xnames+ynames, sim=sim, snap=snap, N=len(sdss.y), inp=inp, sampling=str(sampling))
    plot_data.hist_data(('sdss-train', 'sdss-test', 'eagle-total'), [np.hstack((x_train, y_train)), np.hstack((x_test, y_test)), np.hstack((eagle.x, eagle.y))], edges, xlim=[-1.5,1.5], ylim=[-1.1,1.1])
    #plot_data.hist_data(('eagle', 'sdss-total'), [np.vstack((np.hstack((x_train, y_train)), np.hstack((x_test, y_test)))), np.hstack((sdss.x, sdss.y))], edges, xlim=[-1.5,1.5], ylim=[-1.1,1.1])
    plot_data.datanames = xnames+[ynames[0].split('(')[0]+'$']
    plot_data.statistical_matrix(x_test, y_test, ['eagle'], simple=True)
    plot_data.statistical_matrix(eagle.x, eagle.y, ['sdss'], simple=True)

input_size = len(x_train[0])
output_size = len(y_train[0])
nodes = [input_size] + h_nodes + [output_size]
print("nodes in the network: ", nodes)

#read hyperparameters
nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, epochs, batch_size, optimizer)
new_result = AdditionalValidationSets([(eagle.x, eagle.y, 'val2')])
callbacks = [new_result]


#------------------------------------------Cross-validate on SDSS--------------------------------------------------------------
K = 5

#crossvalidation data
X_train, Y_train, X_val, Y_val = cross_validation(x_train, y_train, K, seed)

cv_score = np.empty((K, 5))
for i in range(K):
    #make architecture of the network
    nn.MLP_model()
    result = nn.model.fit(X_train[i], Y_train[i], batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_val[i], Y_val[i]), callbacks=callbacks)
    cv_score[i,:2] = nn.model.evaluate(X_val[i], Y_val[i], verbose=0)
    y_pred = nn.model.predict(X_val[i]) #predict
    backend.clear_session()
    cv_score[i] = add_scores(cv_score[i,:2], X_val[i], Y_val[i], y_pred) #add scores
avg_cv_score = np.average(cv_score, axis=0)

print("Cross-validated errors (EAGLE): MAE = %.3e ; R^2 = %.3f ; adjusted R^2 = %.3f ; error_mean = %.3e ; error_sigma = %.3e"%tuple(avg_cv_score))


#------------------------------------------Train and test on SDSS--------------------------------------------------------------
#train, evaluate, predict, postprocess
nn.MLP_model()
result = nn.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test), callbacks=callbacks) #train
score = nn.model.evaluate(x_test, y_test, verbose=0) #evaluate
y_pred = nn.model.predict(x_test) #predict
score = add_scores(score, x_test, y_test, y_pred) #add scores
print("Errors (EAGLE): MAE = %.3e ; R^2 = %.3f ; adjusted R^2 = %.3f ; error_mean = %.3e ; error_sigma = %.3e"%tuple(score))


#------------------------------------------Compare with EAGLE--------------------------------------------------------------
#evaluate, predict, postprocess
eagle_score = nn.model.evaluate(eagle.x, eagle.y, verbose=0) #evaluate
eagle.ypred = nn.model.predict(eagle.x) #predict
eagle_score = add_scores(eagle_score, eagle.x, eagle.y, eagle.ypred)
print("Errors (SDSS): MAE = %.3e ; R^2 = %.3g ; adjusted R^2 = %.3g ; error_mean = %.3e ; error_sigma = %.3e"%tuple(eagle_score))

np.save('/disks/strw9/vanweenen/mrp2/plots/score_'+xcols_string+'_'+eagle_ycols[0]+'_inp='+inp+'_sampling='+str(sampling)+'.pdf', np.vstack((avg_cv_score, score, eagle_score)))


#------------------------------------------Calculate shapley values--------------------------------------------------------------
explainer = shap.DeepExplainer(nn.model, x_train)
shap_sdss = explainer.shap_values(x_test)[0]
shap_eagle = explainer.shap_values(eagle.x)[0]

shapdir = '/disks/strw9/vanweenen/mrp2/plots/mlp_dustyflux_mstar/feature_importance/shap/tests_linearfunc/one_coeff/'

#np.save(shapdir + 'shap_eagle_' + inp + '.npy', shap_eagle)
#np.save(shapdir + 'shap_sdss_' + inp + '.npy', shap_sdss)

#shap_eagle_ref = np.load(shapdir + 'shap_eagle_' + inp + '.npy')
#shap_sdss_ref = np.load(shapdir + 'shap_sdss_' + inp + '.npy')
#shap_eagle -= shap_eagle_ref
#shap_sdss -= shap_sdss_ref

plot_shap = PLOT_SHAP(xnames, xcols_string, inp, str(sampling))
plot_shap.summary_plot(x_test, shap_sdss, 'sdss', shap_eagle, 'eagle', plottype = 'bar')
plot_shap.summary_plot(x_test, shap_sdss, 'sdss', plottype = 'dot', dotsize=10)


#------------------------------------------Plotting--------------------------------------------------------------
x_test, y_test, y_pred = sdss.postprocess(eagle, x_test, y_test, y_pred) #postprocessing #TODO

#plot
if plotting:
    plot = PLOT_NN(eagle, nn, xnames, ynames, dataname='sdss', N=len(y_test), xcols_string=xcols_string, inp=inp, sampling=str(sampling), score=score)#[mse[19], r2[19]]
    plot.plot_learning_curve(new_result)

    plot.plot_input_output(x_test, y_test, y_pred, 'scatter') #scatter, contour
    plot.plot_true_predict(y_test, y_pred, 'scatter', cs_mask) #scatter, hexbin, contour, contourf
    plot.plot_output_error(y_test, y_pred, 'contourf', cs_mask) #scatter, hexbin, contour

eagle.x, eagle.y, eagle.ypred = eagle.postprocess(eagle.x, eagle.y, eagle.ypred) #postprocessing #todo

#plot
if plotting:
    plot = PLOT_NN(eagle, nn, xnames, ynames, dataname='eagle', N=len(eagle.y), xcols_string=xcols_string, inp=inp, sampling=str(sampling), score=eagle_score)#[mse[19], r2[19]]
    plot.plot_input_output(eagle.x, eagle.y, eagle.ypred) #scatter, contour
    plot.plot_true_predict(eagle.y, eagle.ypred, 'scatter') #scatter, hexbin, contour, contourf
    plot.plot_output_error(eagle.y, eagle.ypred, 'contourf', ylim=(-0.3, 0.15)) #scatter, hexbin, contour


