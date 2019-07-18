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
args = parser.parse_args()
assert args.input == 'nocolors' or args.input == 'subsetcolors' or args.input == 'allcolors'

seed = 7
np.random.seed(seed) #fix seed for reproducibility
tf.set_random_seed(seed)

plotting = False

inp = args.input#'nocolors'#'subsetcolors'#'allcolors'#

#EAGLE settings
fol = ''
sim = 'RefL0100N1504'   # simulation
cat = 'sdss'            # catalogue to build model to
dust = 'dusty'          # '' if no dust
snap = 27               # snapshot to use
redshift = 1.0063854e-01#0.1# redshift of snapshot 
#redshift_array = np.flip(np.array([2.2204460e-16, 1.0063854e-01, 1.8270987e-01, 2.7090108e-01, 3.6566857e-01, 5.0310730e-01, 6.1518980e-01, 7.3562960e-01, 8.6505055e-01, 1.0041217e+00, 1.2593315e+00, 1.4867073e+00, 1.7369658e+00, 2.0124102e+00, 2.2370370e+00, 2.4784133e+00, 3.0165045e+00, 3.5279765e+00, 3.9836636e+00, 4.4852138e+00, 5.0372367e+00, 5.4874153e+00, 5.9711623e+00, 7.0495663e+00, 8.0746160e+00, 8.9878750e+00, 9.9930330e+00]))
#redshift = redshift_array[snap - 2]

#preprocessing
cs = False ; cs_mask = None #central satellite analysis (only change cs)
sampling = 'random' ; bins = 10 ; count = 50 ; N = 625 ; scaling = False if sampling == 'uniform' else True #both same uniform mass distribution

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

def extract_flux_color(inp):
    f, c = [], []
    for i in inp:
        if len(i) == 1:
            f.append(i)
        elif len(i) == 2:
            c.append(i) 
    return tuple(f), tuple(c)

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

   
def mlp(seed, plotting, inp, fol, sim, cat, dust, snap, redshift, cs, cs_mask, sampling, bins, count, N, scaling, h_nodes, activation, optimizer, dropout, perc_train, loss, epochs, batch_size, fluxes, colors):
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

    #------------------------------------------ Read data --------------------------------------------------------------
    #read data
    eagle = EAGLE(fol, sim, cat, dust, snap, redshift, seed)
    x_train, y_train, x_test, y_test = eagle.preprocess(colors, eagle_xcols, eagle_ycols, eagle_xtype, sampling, bins, count, N, scaling=scaling)

    #sdss
    sdss = SDSS(sdss_xcols, sdss_ycols, sdss_xtype, redshift)
    sdss.preprocess(eagle, colors, sampling, int(N*0.2), scaling=scaling)

    print("EAGLE: len(eagle.x) = %s ; len(x_train) = %s ; len(x_test) = %s ; len(sdss.x) = %s"%(len(eagle.x), len(x_train), len(x_test), len(sdss.x)))

    if sampling == 'uniform':
        uniform_mass_sampling(eagle, ref_pre=sdss, count=100, cs_arr=True)
        uniform_mass_sampling(sdss, ref_post=eagle, count=int(100*0.2), cs_arr=False) #TODO
        random_sampling(eagle, N)
        random_sampling(sdss, int(N*0.2), cs_arr=False) #TODO
        eagle.scaling()
        sdss.scaling(eagle)
        x_train, x_test, y_train, y_test, eagle.cs_train, eagle.cs_test = train_test_split(eagle.x, eagle.y, eagle.cs, test_size=1-perc_train, random_state=seed, shuffle=True)
        print("EAGLE: len(eagle.x) = %s ; len(x_train) = %s ; len(x_test) = %s ; len(sdss.x) = %s"%(len(eagle.x), len(x_train), len(x_test), len(sdss.x)))

    input_size = len(x_train[0])
    output_size = len(y_train[0])
    nodes = [input_size] + h_nodes + [output_size]
    print("nodes in the network: ", nodes)

    #read hyperparameters
    nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, epochs, batch_size, optimizer)

    #------------------------------------------Cross-validate on EAGLE--------------------------------------------------------------
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
        backend.clear_session()
    avg_cv_score = np.average(cv_score, axis=0)

    print("Cross-validated errors (EAGLE): MAE = %.3e ; R^2 = %.3f ; adjusted R^2 = %.3f ; error_mean = %.3e ; error_sigma = %.3e"%tuple(avg_cv_score)) 
    """    
    #------------------------------------------Train and test on EAGLE--------------------------------------------------------------
    #train, evaluate, predict, postprocess
    nn.MLP_model()
    result = nn.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test))#train
    score = nn.model.evaluate(x_test, y_test, verbose=0) #evaluate
    y_pred = nn.model.predict(x_test) #predict
    score = add_scores(score, x_test, y_test, y_pred) #add scores
    print("Errors (EAGLE): MAE = %.3e ; R^2 = %.3f ; adjusted R^2 = %.3f ; error_mean = %.3e ; error_sigma = %.3e"%tuple(score))

    #------------------------------------------Compare with SDSS--------------------------------------------------------------
    #evaluate, predict, postprocess
    sdss_score = nn.model.evaluate(sdss.x, sdss.y, verbose=0) #evaluate
    sdss.ypred = nn.model.predict(sdss.x) #predict
    sdss_score = add_scores(sdss_score, sdss.x, sdss.y, sdss.ypred)
    print("Errors (SDSS): MAE = %.3e ; R^2 = %.3g ; adjusted R^2 = %.3g ; error_mean = %.3e ; error_sigma = %.3e"%tuple(sdss_score))
    """
    backend.clear_session()
    del(eagle, sdss, nn)
    return avg_cv_score[0]

inps = (('u',), ('g',), ('r',), ('i',), ('z',), ('ug',), ('ur',), ('ui',), ('uz',), ('gr',), ('gi',), ('gz',), ('ri',), ('rz',), ('iz',))

tot_save_arr = []
inps_poss = inps
for n in range(15):
    errors_arr = []
    for i in inps_poss:
        fluxes, colors = extract_flux_color(i)
        print(fluxes, colors)
        
        cv_mae = mlp(seed, plotting, inp, fol, sim, cat, dust, snap, redshift, cs, cs_mask, sampling, bins, count, N, scaling, h_nodes, activation, optimizer, dropout, perc_train, loss, epochs, batch_size, fluxes, colors)
        errors_arr.append(cv_mae)
        
    order = np.argsort(errors_arr)
    inps_remain = (inps_poss[order[0]], inps_poss[order[1]], inps_poss[order[2]])
    print("Best inputs for ", n+1, " nodes:", inps_remain)
    
    errors_arr = np.array(errors_arr).reshape(-1,1)
    inps_poss = np.array(inps_poss)
    
    save_arr = np.hstack((inps_poss, np.tile([''], (inps_poss.shape[0],15-inps_poss.shape[1])), errors_arr))
    tot_save_arr.append(save_arr)
    np.save('/disks/strw9/vanweenen/mrp2/plots/feature_importance/errors_featureimportance_bottomup_'+inp+'_nodes='+str(n+1)+'.pdf', save_arr)
    
    inps_poss = []
    for i in inps_remain:
        for j in inps:
            if j[0] not in i:
                inps_poss.append(i+j)
tot_save_arr = np.concatenate((tot_save_arr), axis=0)
best = np.argsort(tot_save_arr[:,-1].astype(np.float))
print(tot_save_arr[best[0]])
print(tot_save_arr[best[1]])
print(tot_save_arr[best[2]])
np.save('/disks/strw9/vanweenen/mrp2/plots/feature_importance/errors_featureimportance_bottomup_'+inp+'_all.pdf', tot_save_arr)
