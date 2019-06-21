from eagle.io import *
from eagle.plot import *
from eagle.nn import *

from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from keras import backend

from argparse import ArgumentParser 
parser = ArgumentParser()
parser.add_argument("input", help = "Choose nocolors, subsetcolors or allcolors")
args = parser.parse_args()

assert args.input == 'nocolors' or args.input == 'subsetcolors' or args.input == 'allcolors'

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
redshift = 1.0063854e-01#0.1          # redshift of snapshot    

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
xnames=[dust + ' ' + cat + ' ' + i for i in list(fluxes) + [c[0] + '-' + c[1] for c in colors]]
ynames=['$\log_{10} M_{*} (M_{\odot})$']#['$\log_{10}$ SFR $(M_{\odot} yr^{-1})$']#

#preprocessing
sampling = 'uniform' 
#bins = 10 ; count = 100 ; N = 600 #uniform mass distribution
bins = 10 ; count = 125 ; N = 625 #uniform mass distribution


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
    return X_train, X_val, Y_train, Y_val

def add_scores(score, x, y, y_pred):
    y.reshape(len(y),) ; y_pred.reshape(len(y_pred),)
    score[1] = r2_score(y, y_pred)
    return np.append(score, [adjusted_R_squared(score[1], x.shape[1], len(y)), np.mean(y_pred - y), np.var(y_pred - y)])


def uniform_mass_sampling_local(x_train, y_train, cs_train, x_test, y_test, cs_test, sdss, bins=10, count=100, perc_train=0.8):
    y_train = y_train.reshape(-1,1) ; y_test = y_test.reshape(-1,1) ; sdss.y = sdss.y.reshape(-1,1)

    x = (x_train, x_train[cs_train], x_train[~cs_train], x_test, x_test[cs_test], x_test[~cs_test])
    y = (y_train, y_train[cs_train], y_train[~cs_train], y_test, y_test[cs_test], y_test[~cs_test])
    c = (100, 100, 100, 25, 25, 25)
    
    #create histogram for each data set with the same edges
    hist, edges = np.histogram(np.vstack((y_train, y_test)), bins=bins)
    h = []   
    for i in range(len(y)):
        h.append(np.histogram(y[i], bins=edges)[0])
    sdss.hist, sdss.edges = np.histogram(sdss.y, bins=edges)
    
    #bins that remain are the bins for which the count of all datasets is above c[i]
    remain = (hist > count)
    print(len(np.where(remain)[0]))
    for i in range(len(x)):
        remain &= (h[i] > c[i])
        print(len(np.where(remain)[0]))
    remain &= (sdss.hist > c[5])
    print("Final number of bins: ", len(np.where(remain)[0]))

    #array with lower edges and upper edges of bins that are above the minimal count
    #sdss.edges = edges[np.insert(remain, len(hist), False) | np.insert(remain, 0, False)]
    lower_edges = edges[np.insert(remain, len(hist), False)]
    upper_edges = edges[np.insert(remain, 0, False)]
    for i in range(len(h)):
        h[i] = h[i][remain]
    sdss.hist = sdss.hist[remain]

    #create new x and y data consisting of an equal number of galaxies per mass bin
    #for each bin, randomly shuffle the data in the bin, and then keep the first count galaxies (perm)
    x_equal = [[],[],[],[],[],[]]
    y_equal = [[],[],[],[],[],[]]
    sdss.x_equal = np.empty((c[5]*len(sdss.hist), sdss.x.shape[1]))
    sdss.y_equal = np.empty((c[5]*len(sdss.hist), sdss.y.shape[1]))
    for i in range(len(x)):
        x_equal[i] = np.empty((c[i]*len(h[i]), x[i].shape[1]))
        y_equal[i] = np.empty((c[i]*len(h[i]), y[i].shape[1]))
        for j in range(len(lower_edges)):
            perm = np.random.permutation(h[i][j])[:c[i]]
            mask = (y[i] >= lower_edges[j]) & (y[i] < upper_edges[j])
            mask = mask.reshape(len(mask),)
            x_equal[i][j*c[i]:(j+1)*c[i]] = x[i][mask][perm]
            y_equal[i][j*c[i]:(j+1)*c[i]] = y[i][mask][perm]
        x_equal[i], y_equal[i] = shuffle(x_equal[i], y_equal[i])
    for j in range(len(lower_edges)):
        perm = np.random.permutation(sdss.hist[j])[:c[5]]
        mask = (sdss.y >= lower_edges[j]) & (sdss.y < upper_edges[j])
        mask = mask.reshape(len(mask),)
        sdss.x_equal[j*c[5]:(j+1)*c[5]] = sdss.x[mask][perm]
        sdss.y_equal[j*c[5]:(j+1)*c[5]] = sdss.y[mask][perm]        
    return x_equal, y_equal

#------------------------------------------ Read data --------------------------------------------------------------
#read data
eagle = EAGLE(fol, sim, cat, dust, snap, redshift, seed, eagle_xcols, eagle_ycols, eagle_xtype)
eagle.preprocess(colors)

#sdss
sdss = SDSS(sdss_xcols, sdss_ycols, sdss_xtype, redshift)
sdss.preprocess(colors)

#divide into train and test set
x_train, x_test, y_train, y_test, eagle.cs_train, eagle.cs_test = train_test_split(eagle.x, eagle.y, eagle.cs, test_size=1-perc_train, random_state=seed, shuffle=True)
print("Total size of data: %s; size of training set: %s ; size of test set: %s"%(len(eagle.x), len(x_train), len(x_test)))
print("EAGLE: len(eagle.x) = %s ; len(x_train) = %s ; len(x_test) = %s ; len(sdss.x) = %s"%(len(eagle.x), len(x_train), len(x_test), len(sdss.x)))

#uniform mass sampling
x_equal, y_equal = uniform_mass_sampling_local(x_train, y_train, eagle.cs_train, x_test, y_test, eagle.cs_test, sdss)

#scaling
eagle.xscaler = MinMaxScaler(feature_range=(-1,1)).fit(np.vstack(x_equal))
eagle.yscaler = MinMaxScaler(feature_range=(-1,1)).fit(np.vstack(y_equal))
for i in range(len(x_equal)):
    x_equal[i] = eagle.xscaler.transform(x_equal[i])
    y_equal[i] = eagle.yscaler.transform(y_equal[i])
sdss.x, sdss.y = sdss.x_equal, sdss.y_equal
sdss.scaling(eagle)

x_train, x_train_c, x_train_s, x_test, x_test_c, x_test_s = x_equal
y_train, y_train_c, y_train_s, y_test, y_test_c, y_test_s = y_equal

print("EAGLE: train = %s ; train_c = %s ; train_s = %s ; test = %s ; test_c = %s ; test_s = %s ; len(sdss.x) = %s"%(len(x_train), len(x_train_c), len(x_train_s), len(x_test), len(x_test_c), len(x_test_s), len(sdss.x)))

if plotting:
    plot_data = PLOT_DATA(xnames+ynames, sim=sim, cat=cat, snap=snap)
    plot_data.hist_data(('eagle-train', 'eagle-test', 'sdss-total'), [np.hstack((x_train, y_train)), np.hstack((x_test, y_test)), np.hstack((sdss.x, sdss.y))], eagle.yscaler.transform(sdss.edges.reshape(-1,1)).T[0], xlim=[-1.5,1.1], ylim=[-1.1,1.1])
plot_data.hist_data(('eagle-train', 'eagle-test', 'eagle-train-centrals', 'eagle-train-satellites', 'eagle-test-centrals', 'eagle-test-satellites', 'sdss-total'), [np.hstack((x_train, y_train)), np.hstack((x_test, y_test)), np.hstack((x_train_c, y_train_c)), np.hstack((x_train_s, y_train_s)), np.hstack((x_test_c, y_test_c)), np.hstack((x_test_s, y_test_s)), np.hstack((sdss.x, sdss.y))], eagle.yscaler.transform(sdss.edges.reshape(-1,1)).T[0], xlim=[-1.5,1.1], ylim=[-1.1,1.1])

input_size = len(x_train[0])
output_size = len(y_train[0])
nodes = [input_size] + h_nodes + [output_size]
print("nodes in the network: ", nodes)

#read hyperparameters 
nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, epochs, batch_size, optimizer)
new_result = AdditionalValidationSets([(sdss.x, sdss.y, 'val2'), (x_test_c, y_test_c, 'val3'), (x_test_s, y_test_s, 'val4')])
callbacks = [new_result]

#------------------------------------------Cross-validate on EAGLE--------------------------------------------------------------
K = 5

#crossvalidation data
X_train, X_val, Y_train, Y_val = cross_validation(x_train_s, y_train_s, K, seed) #TODO

cv_score = np.empty((K, 5))
for i in range(K):
    #make architecture of the network
    nn.MLP_model()
    result = nn.model.fit(X_train[i], Y_train[i], batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_val[i], Y_val[i]), callbacks=callbacks)
    cv_score[i,:2] = nn.model.evaluate(X_val[i], Y_val[i], verbose=0)
    y_pred = nn.model.predict(X_val[i]) #predict
    cv_score[i] = add_scores(cv_score[i,:2], X_val[i], Y_val[i], y_pred) #add scores
avg_cv_score = np.average(cv_score, axis=0)
print("Cross-validated errors (EAGLE): MAE = %.3e ; R^2 = %.3f ; adjusted R^2 = %.3f ; error_mean = %.3e ; error_sigma = %.3e"%tuple(avg_cv_score))


#------------------------------------------Train and test on EAGLE--------------------------------------------------------------
#train, evaluate, predict, postprocess
nn.MLP_model()
result = nn.model.fit(x_train_s, y_train_s, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test), callbacks=callbacks) #train #TODO

#evaluate
score = nn.model.evaluate(x_test, y_test, verbose=0) 
score_c = nn.model.evaluate(x_test_c, y_test_c, verbose=0)
score_s = nn.model.evaluate(x_test_s, y_test_s, verbose=0)
#predict
y_pred = nn.model.predict(x_test) 
y_pred_c = nn.model.predict(x_test_c)
y_pred_s = nn.model.predict(x_test_s)
#add scores
score = add_scores(score, x_test, y_test, y_pred)
score_c = add_scores(score_c, x_test_c, y_test_c, y_pred_c)
score_s = add_scores(score_s, x_test_s, y_test_s, y_pred_s)
print("Errors (EAGLE): MAE = %.3e ; R^2 = %.3f ; adjusted R^2 = %.3f ; error_mean = %.3e ; error_sigma = %.3e"%tuple(score))
print("Errors (EAGLE) centrals: MAE = %.3e ; R^2 = %.3g ; adjusted R^2 = %.3g ; error_mean = %.3e ; error_sigma = %.3e"%tuple(score_c))
print("Errors (EAGLE) satellites: MAE = %.3e ; R^2 = %.3g ; adjusted R^2 = %.3g ; error_mean = %.3e ; error_sigma = %.3e"%tuple(score_s))

#------------------------------------------Compare with SDSS--------------------------------------------------------------
#evaluate, predict, postprocess
sdss_score = nn.model.evaluate(sdss.x, sdss.y, verbose=0) #evaluate
sdss.ypred = nn.model.predict(sdss.x) #predict
sdss_score = add_scores(sdss_score, sdss.x, sdss.y, sdss.ypred)
print("Errors (SDSS): MAE = %.3e ; R^2 = %.3g ; adjusted R^2 = %.3g ; error_mean = %.3e ; error_sigma = %.3e"%tuple(sdss_score))

np.save('/disks/strw9/vanweenen/mrp2/plots/score_CS_'+eagle_ycols[0]+'_inp='+inp+'_sampling='+str(sampling)+'.pdf', np.vstack((avg_cv_score, score, score_c, score_s, sdss_score)))

#------------------------------------------Plotting--------------------------------------------------------------
#postprocessing
x_test, y_test, y_pred = eagle.postprocess(x_test, y_test, y_pred) 
x_test_c, y_test_c, y_pred_c = eagle.postprocess(x_test_c, y_test_c, y_pred_c) 
x_test_s, y_test_s, y_pred_s = eagle.postprocess(x_test_s, y_test_s, y_pred_s) 

#plot
if plotting:
    plot = PLOT_NN(eagle, nn, xnames, ynames, dataname='eagle', N=len(y_test), xcols_string=xcols_string, inp=inp, sampling=str(sampling), score=score_s)#[mse[19], r2[19]] #TODO
    plot.plot_learning_curve(new_result)
    plot.plot_input_output(x_test, y_test, y_pred, 'scatter') #scatter, contour
    plot.plot_true_predict(y_test, y_pred, 'scatter', True, y_test_c, y_pred_c, y_test_s, y_pred_s) #scatter, hexbin, contour(f)
    plot.plot_output_error(y_test, y_pred, 'contour', True, y_test_c, y_pred_c, y_test_s, y_pred_s) #scatter, hexbin, contour(f)

sdss.x, sdss.y, sdss.ypred = sdss.postprocess(eagle, sdss.x, sdss.y, sdss.ypred) #postprocessing

#plot
if plotting:
    plot = PLOT_NN(eagle, nn, xnames, ynames, dataname='sdss', N=len(sdss.y), xcols_string=xcols_string, inp=inp, sampling=str(sampling), score=sdss_score)#[mse[19], r2[19]]
    plot.plot_input_output(sdss.x, sdss.y, sdss.ypred) #scatter, contour
    plot.plot_true_predict(sdss.y, sdss.ypred, 'scatter') #scatter, hexbin, contour, contourf
    plot.plot_output_error(sdss.y, sdss.ypred, 'contourf', bins=5, ylim=(-.1, .5)) #scatter, hexbin, contour
