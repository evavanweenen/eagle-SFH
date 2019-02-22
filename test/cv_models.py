from eagle.io import *
from eagle.nn import *
import numpy as np
from sklearn.model_selection import KFold

seed = 7
np.random.seed(seed) #fix seed for reproducibility

path =  '/disks/strw9/vanweenen/mrp2/optimization/'

K = 10

#EAGLE settings
fol = ''
sim = 'RefL0100N1504' #simulation
cat = 'dusty-sdss' #catalogue to build model to
snap = 27 #snapshot to use
redshift = 0.1

#eagle
eagle_xtype = 'flux'
eagle_xcols = ['dusty' + eagle_xtype + '_sdss_' + f for f in ['u','g','r','i','z']]
eagle_ycols = ['m_star']#['sfr']

#hyperparams
perc_train = .8
h_nodes = [50, 50, 40]
dropout = [0, 0, 0] 
activation = ['linear', 'relu', 'tanh', 'linear']
loss = 'mean_squared_error'
lr_rate = 0.001
epochs = 15
batch_size = 128
optimizer = 'Adam'

def cross_validation(X, Y, K, seed):
    kf = KFold(n_splits=K, random_state=seed)
    X_train = [] ; Y_train = []; X_val = []; Y_val = []
    for idx_train, idx_val in kf.split(X):
        X_train.append(X[idx_train])
        Y_train.append(Y[idx_train])
        X_val.append(X[idx_val])
        Y_val.append(Y[idx_val])       
    return X_train, Y_train, X_val, Y_val

#read eagle data
eagle = EAGLE(fol, sim, cat, snap, redshift, eagle_xcols, eagle_ycols, perc_train)
x_train, y_train, x_test, y_test = eagle.preprocess(eagle_xtype)

input_size = len(x_train[0])
output_size = len(y_train[0])

#crossvalidation data
X_train, Y_train, X_val, Y_val = cross_validation(x_train, y_train, K, seed)

#read hyperparameters 
nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, lr_rate, epochs, batch_size, optimizer)

cv_score = np.empty((K, 2))
for i in range(K):
      #make architecture of the network
      nn.MLP_model()
      result = nn.model.fit(X_train[i], Y_train[i], batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_val[i], Y_val[i]))
      cv_score[i] = nn.model.evaluate(X_val[i], Y_val[i], verbose=0)
avg_cv_score = np.average(cv_score, axis=0)

print("Errors (EAGLE): MSE = %.4e ; R^2 = %.4g"%(avg_cv_score[0], avg_cv_score[1]))



