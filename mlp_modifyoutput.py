from eagle.io import *
from eagle.plot import *
from eagle.nn import *

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
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

fluxes = ('u', 'g', 'r', 'i', 'z')
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
sampling = args.sampling#'uniform'
N=625
#bins = 10 ; count = 50 ; N = 625 ; scaling = True if sampling == None else False #both same uniform mass distribution
#bins=10 ; count=125 ; N=625
#bins=200 ; count=5 ; N=625

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
Note that we perform everything in logscale since the errors are also given in log scale!
"""

#-----------------EAGLE preprocessing-------------------
eagle = EAGLE(fol, sim, cat, dust, snap, redshift, seed, eagle_xcols, eagle_ycols, eagle_xtype)
eagle.preprocess(colors)

#-----------------SDSS preprocessing-------------------
sdss = SDSS(sdss_xcols, sdss_ycols, sdss_xtype, redshift)

#datacols
prefix = sdss.ycols[0].replace('median', '')
sdss.datacols += [prefix + '16th', prefix + '84th']

#read data
sdss.read_data()

#select only galaxies of given redshift
sdss.select_redshift(frac=5e-3)

#to magnitude
if sdss.xtype == 'flux':
    to_magnitude(sdss, sdss.xtype, luminosity_distance_sdss)

#add colors
add_colors(sdss, sdss.xtype+'_', colors)

#remove galaxies with nan mass
sdss.data = sdss.data[~np.isnan(structured_to_unstructured(sdss.data[sdss.ycols]))]

#-----------------Add variance to mass-------------------
#array of mass errors
mass_sigma = (sdss.data[prefix + '84th'] - sdss.data[prefix + '16th'])/2

#mass arrays
eagle_mass = structured_to_unstructured(eagle.data[eagle.ycols])
sdss_mass = structured_to_unstructured(sdss.data[sdss.ycols])

#mass_bins = 20 ; mass_count = 25
mass_bins = 300; mass_count = 5

#mass errors arrays
mass_errors = np.empty((mass_bins))

#histogram of sdss logarithmic stellar mass
hist_sdss, mass_edges = np.histogram(sdss_mass, bins=mass_bins)

#histogram of eagle logarithmic stellar mass
hist_eagle = np.histogram(eagle_mass, bins=mass_edges)[0]

#remaining bins
remain = (hist_sdss > mass_count) & (hist_eagle > mass_count)
remain = consecutive_trues(remain)

print("Number of remaining bins: ", len(np.where(remain)[0]))
for j in range(mass_bins):
    mask_eagle = (eagle_mass >= mass_edges[j]) & (eagle_mass < mass_edges[j+1])
    mask_sdss = (sdss_mass >= mass_edges[j]) & (sdss_mass < mass_edges[j+1])
    if remain[j]:
        mass_errors[j] = np.average(mass_sigma[mask_sdss])
        eagle.data[eagle.ycols][mask_eagle] = np.random.normal(structured_to_unstructured(eagle.data[eagle.ycols][mask_eagle]), mass_errors[j])
    else:
        eagle.data[eagle.ycols][mask_eagle] = np.nan
        sdss.data[sdss.ycols][mask_sdss] = np.nan
        
#remove nans
eagle.data = eagle.data[~np.isnan(structured_to_unstructured(eagle.data[eagle.ycols]))]
sdss.data = sdss.data[~np.isnan(structured_to_unstructured(sdss.data[sdss.ycols]))]

#eagle remaining preprocessing steps
eagle.x, eagle.y = divide_input_output(eagle)

#sdss remaining preprocessing steps
sdss.datacols = sdss.xcols + sdss.ycols
select_cols(sdss)
sdss.x, sdss.y = divide_input_output(sdss)
sdss.x, sdss.y = remove_zero(sdss.x, sdss.y)
sdss.x, sdss.y = remove_minone(sdss.x, sdss.y)
sdss.y, sdss.y = remove_nan(sdss.x, sdss.y)
sdss.x, sdss.y = remove_inf(sdss.x, sdss.y)

#determine number of bins for sampling
binsize = 0.01046
range_mass = np.amax(eagle.y) - np.amin(eagle.y)
bins = int(range_mass / binsize)
count = int(np.ceil(N / bins / 5)*5)+5
print("number of bins of size ", binsize, " dex: ", bins)
sample(eagle, sdss, sampling, bins, count, N)

eagle.scaling()
sdss.scaling(eagle)

x_train, x_test, y_train, y_test = train_test_split(eagle.x, eagle.y, test_size=0.2, random_state=eagle.seed, shuffle=True)
print("Total size of data: %s; size of training set: %s ; size of test set: %s"%(len(eagle.x), len(x_train), len(x_test)))

print("EAGLE: len(eagle.x) = %s ; len(x_train) = %s ; len(x_test) = %s ; len(sdss.x) = %s"%(len(eagle.x), len(x_train), len(x_test), len(sdss.x)))


if plotting:
    #if sampling == 'uniform':
        #edges = eagle.yscaler.transform(eagle.edges.reshape(-1,1)).T[0]
    #else:
        #edges = 7
    edges = 10
    plot_data = PLOT_DATA(xnames+ynames, sim=sim, snap=snap, N=len(sdss.y), inp=inp, sampling=str(sampling))
    plot_data.hist_data(('eagle-train', 'eagle-test', 'sdss-total'), [np.hstack((x_train, y_train)), np.hstack((x_test, y_test)), np.hstack((sdss.x, sdss.y))], edges, xlim=[-1.5,1.5], ylim=[-1.1,1.1])
    plot_data.datanames = xnames+[ynames[0].split('(')[0]+'$']
    plot_data.statistical_matrix(x_test, y_test, ['eagle'], simple=True)
    plot_data.statistical_matrix(sdss.x, sdss.y, ['sdss'], simple=True)

input_size = len(x_train[0])
output_size = len(y_train[0])
nodes = [input_size] + h_nodes + [output_size]
print("nodes in the network: ", nodes)

#read hyperparameters
nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, epochs, batch_size, optimizer)
new_result = AdditionalValidationSets([(sdss.x, sdss.y, 'val2')])
callbacks = [new_result]

#------------------------------------------Cross-validate on EAGLE--------------------------------------------------------------
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

#------------------------------------------Train and test on EAGLE--------------------------------------------------------------
#train, evaluate, predict, postprocess
nn.MLP_model()
result = nn.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test), callbacks=callbacks) #train
score = nn.model.evaluate(x_test, y_test, verbose=0) #evaluate
y_pred = nn.model.predict(x_test) #predict
score = add_scores(score, x_test, y_test, y_pred) #add scores
print("Errors (EAGLE): MAE = %.3e ; R^2 = %.3f ; adjusted R^2 = %.3f ; error_mean = %.3e ; error_sigma = %.3e"%tuple(score))

#central satellites analysis
if cs == True:
    cs_mask = eagle.cs_test
    score_c = nn.model.evaluate(x_test[cs_mask], y_test[cs_mask], verbose=0)
    score_s = nn.model.evaluate(x_test[~cs_mask], y_test[~cs_mask], verbose=0)
    score_c = add_scores(score_c, x_test[cs_mask], y_test[cs_mask], y_pred[cs_mask])
    score_s = add_scores(score_s, x_test[~cs_mask], y_test[~cs_mask], y_pred[~cs_mask])
    print("Errors (EAGLE) centrals: MAE = %.3e ; R^2 = %.3g ; adjusted R^2 = %.3g ; error_mean = %.3e ; error_sigma = %.3e"%tuple(score_c))
    print("Errors (EAGLE) satellites: MAE = %.3e ; R^2 = %.3g ; adjusted R^2 = %.3g ; error_mean = %.3e ; error_sigma = %.3e"%tuple(score_s))

#------------------------------------------Compare with SDSS--------------------------------------------------------------
#evaluate, predict, postprocess
sdss_score = nn.model.evaluate(sdss.x, sdss.y, verbose=0) #evaluate
sdss.ypred = nn.model.predict(sdss.x) #predict
sdss_score = add_scores(sdss_score, sdss.x, sdss.y, sdss.ypred)
print("Errors (SDSS): MAE = %.3e ; R^2 = %.3g ; adjusted R^2 = %.3g ; error_mean = %.3e ; error_sigma = %.3e"%tuple(sdss_score))

np.save('/disks/strw9/vanweenen/mrp2/plots/score_'+xcols_string+'_'+eagle_ycols[0]+'_inp='+inp+'_sampling='+str(sampling)+'.pdf', np.vstack((avg_cv_score, score, sdss_score)))

#------------------------------------------Calculate shapley values--------------------------------------------------------------
explainer = shap.DeepExplainer(nn.model, x_train)
shap_eagle = explainer.shap_values(x_test)[0]
shap_sdss = explainer.shap_values(sdss.x)[0]

shapdir = '/disks/strw9/vanweenen/mrp2/plots/mlp_dustyflux_mstar/feature_importance/shap/tests_linearfunc/one_coeff/'

#np.save(shapdir + 'shap_eagle_' + inp + '.npy', shap_eagle)
#np.save(shapdir + 'shap_sdss_' + inp + '.npy', shap_sdss)

#shap_eagle_ref = np.load(shapdir + 'shap_eagle_' + inp + '.npy')
#shap_sdss_ref = np.load(shapdir + 'shap_sdss_' + inp + '.npy')
#shap_eagle -= shap_eagle_ref
#shap_sdss -= shap_sdss_ref

plot_shap = PLOT_SHAP(xnames, xcols_string, inp, str(sampling))
plot_shap.summary_plot(x_test, shap_eagle, 'eagle', shap_sdss, 'sdss', plottype = 'bar')
plot_shap.summary_plot(x_test, shap_eagle, 'eagle', plottype = 'dot', dotsize=10)

#------------------------------------------Plotting--------------------------------------------------------------
x_test, y_test, y_pred = eagle.postprocess(x_test, y_test, y_pred) #postprocessing

#plot
if plotting:
    plot = PLOT_NN(eagle, nn, xnames, ynames, dataname='eagle', N=len(y_test), xcols_string=xcols_string, inp=inp, sampling=str(sampling), score=score)#[mse[19], r2[19]]
    plot.plot_learning_curve(new_result)

    plot.plot_input_output(x_test, y_test, y_pred, 'scatter') #scatter, contour
    plot.plot_true_predict(y_test, y_pred, 'scatter', cs_mask) #scatter, hexbin, contour, contourf
    plot.plot_output_error(y_test, y_pred, 'contourf', cs_mask) #scatter, hexbin, contour

sdss.x, sdss.y, sdss.ypred = sdss.postprocess(eagle, sdss.x, sdss.y, sdss.ypred) #postprocessing

#plot
if plotting:
    plot = PLOT_NN(eagle, nn, xnames, ynames, dataname='sdss', N=len(sdss.y), xcols_string=xcols_string, inp=inp, sampling=str(sampling), score=sdss_score)#[mse[19], r2[19]]
    plot.plot_input_output(sdss.x, sdss.y, sdss.ypred) #scatter, contour
    plot.plot_true_predict(sdss.y, sdss.ypred, 'scatter') #scatter, hexbin, contour, contourf
    plot.plot_output_error(sdss.y, sdss.ypred, 'contourf', ylim=(-0.15, 0.3)) #scatter, hexbin, contour


"""
#------------------------------------------CREATE GIF--------------------------------------------------------------
#read data
eagle = EAGLE(fol, sim, cat, snap, redshift, eagle_xcols, eagle_ycols, perc_train)
x_train, y_train, x_test, y_test = eagle.preprocess(eagle_xtype)

input_size = len(x_train[0])
output_size = len(y_train[0])
nodes = [input_size] + h_nodes + [output_size]
print("nodes in the network: ", nodes)

#read hyperparameters and make network
nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, epochs, batch_size, optimizer)
nn.MLP_model()

Y_pred, mse, r2 = [], [], [] 
for i in range(epochs):
    #train, evaluate, predict, postprocess
    result = nn.model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0, validation_data=(x_test, y_test)) #train 
    score = nn.model.evaluate(x_test, y_test, verbose=0) #evaluate
    y_pred = nn.model.predict(x_test) #predict
    x_test_log, y_test_log, y_pred = eagle.postprocess(x_test, y_test, y_pred) #postprocess
    #plotting
    plot = PLOT_NN(eagle, nn, xnames, ynames, MLtype='MLP', score=score, epochs=i)
    #plot.plot_input_output(x_test_log, y_test_log, y_pred)
    #plot.plot_true_predict(y_test_log, y_pred)
    #save values for gif    
    Y_pred.append(y_pred) ; mse.append(score[0]) ; r2.append(score[1])

#plot gif
plot.gif_input_output(x_test_log, y_test_log, Y_pred=Y_pred, mse=mse, r2=r2)
"""

