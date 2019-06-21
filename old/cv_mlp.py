from eagle.io import *
from eagle.nn import *
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf

seed = 7
np.random.seed(seed) #fix seed for reproducibility
tf.set_random_seed(seed)

path =  '/disks/strw9/vanweenen/mrp2/optimization/'

K = 5

#EAGLE settings
fol = ''
sim = 'RefL0100N1504'   # simulation
cat = 'sdss'            # catalogue to build model to
dust = 'dusty'          # '' if no dust
snap = 27               # snapshot to use
redshift = 0.1          # redshift of snapshot    

fluxes = ('u', 'g', 'r', 'i', 'z')
colors = ()#('ug', 'gr', 'ri', 'iz')#('ug', 'ur', 'ui', 'uz', 'gr', 'gi', 'gz', 'ri', 'rz', 'iz')#
xcols = fluxes + colors

#eagle
eagle_xtype = 'flux'
eagle_xcols = [dust + eagle_xtype + '_' + cat + '_' + f for f in xcols]
eagle_ycols = ['m_star']#['sfr']#

#sdss
sdss_xtype = 'flux'
sdss_xcols = [sdss_xtype + '_' + f for f in xcols]
sdss_ycols = ['Mstellar_median']#['SFR_median']#

#preprocessing
uniform_mass = False ; bins = 10 ; count = 100 #uniform mass distribution
random_sample = False ; N = 900 #random sample

#Hyperas no colors
h_nodes = [60, 40, 40]
activation = ['tanh', 'relu', 'linear', 'linear']
optimizer = 'Adam'
"""
#Hyperas subset colors
h_nodes = [60, 50, 60]
activation = ['linear', 'relu', 'tanh', 'linear']
optimizer = 'Adam'

#Hyperas all colors
h_nodes = [60, 50, 60]
activation = ['linear', 'relu', 'relu', 'linear']
optimizer = 'Adam'
"""
dropout = [0, 0, 0]
loss = 'mean_squared_error'
perc_train = .8
epochs = 15
batch_size = 128

def add_scores(score, x, y, y_pred):
    y.reshape(len(y),) ; y_pred.reshape(len(y_pred),)
    #score[1] = r2_score(y_test, y_pred)
    score = np.append(score, [adjusted_R_squared(score[1], x.shape[1], len(y)), np.mean(y_pred - y), np.var(y_pred - y)])
    return score

def cross_validation(X, Y, K, seed):
    kf = KFold(n_splits=K, random_state=seed)
    X_train = [] ; Y_train = []; X_val = []; Y_val = []
    for idx_train, idx_val in kf.split(X):
        X_train.append(X[idx_train])
        Y_train.append(Y[idx_train])
        X_val.append(X[idx_val])
        Y_val.append(Y[idx_val])       
    return X_train, Y_train, X_val, Y_val

#------------------------------------------Cross-validate on EAGLE--------------------------------------------------------------
#read data
eagle = EAGLE(fol, sim, cat, dust, snap, redshift, seed)
x_train, y_train, x_test, y_test = eagle.preprocess(colors, eagle_xcols, eagle_ycols, eagle_xtype, uniform_mass, bins, count, random_sample, N)

input_size = len(x_train[0])
output_size = len(y_train[0])

#crossvalidation data
X_train, Y_train, X_val, Y_val = cross_validation(x_train, y_train, K, seed)

#read hyperparameters 
nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, epochs, batch_size, optimizer)

cv_score = np.empty((K, 5))
for i in range(K):
      #make architecture of the network
      nn.MLP_model()
      result = nn.model.fit(X_train[i], Y_train[i], batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_val[i], Y_val[i]))
      cv_score[i,:2] = nn.model.evaluate(X_val[i], Y_val[i], verbose=0)
      y_pred = nn.model.predict(X_val[i]) #predict
      cv_score[i] = add_scores(cv_score[i,:2], X_val[i], Y_val[i], y_pred) #add scores
avg_cv_score = np.average(cv_score, axis=0)

print("Cross-validated errors (EAGLE): MSE = %.3e ; R^2 = %.3f ; adjusted R^2 = %.3f ; error_mean = %.3e ; error_sigma = %.3e"%tuple(avg_cv_score))

#------------------------------------------Compare with SDSS--------------------------------------------------------------
#read data
sdss = SDSS(fluxes, sdss_xcols, sdss_ycols, sdss_xtype, redshift)
sdss.preprocess(eagle, colors)

#evaluate, predict, postprocess
sdss_score = nn.model.evaluate(sdss.x, sdss.y, verbose=0) #evaluate
sdss.ypred = nn.model.predict(sdss.x) #predict
sdss_score = add_scores(sdss_score, sdss.x, sdss.y, sdss.ypred)
print("Errors (SDSS): MSE = %.3e ; R^2 = %.3g ; adjusted R^2 = %.3g ; error_mean = %.3e ; error_sigma = %.3e"%tuple(sdss_score))


