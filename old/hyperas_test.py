from eagle.io import *
from eagle.nn import *

import numpy as np
from sklearn.model_selection import KFold
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperas.utils import eval_hyperopt_space

import datetime
import sys
import tensorflow as tf

def cross_validation(X, Y, K, seed):
    kf = KFold(n_splits=K, random_state=seed)
    X_train = [] ; Y_train = []; X_val = []; Y_val = []
    for idx_train, idx_val in kf.split(X):
        X_train.append(X[idx_train])
        Y_train.append(Y[idx_train])
        X_val.append(X[idx_val])
        Y_val.append(Y[idx_val])       
    return X_train, Y_train, X_val, Y_val

def data():
    K = 5 #cross-validation

    seed = 7 ; np.random.seed(seed) #fix seed for reproducibility
    tf.set_random_seed(seed)

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

    cs = False ; cs_mask = None #central satellite analysis (only change cs)
    equal_bins = True #uniform mass distribution

    #read data
    eagle = EAGLE(fol, sim, cat, dust, snap, redshift)
    eagle.read_data()
    x_train, y_train, x_test, y_test = eagle.preprocess(colors, eagle_xcols, eagle_ycols, eagle_xtype, equal_bins)
    
    X_train, Y_train, X_val, Y_val = cross_validation(x_train, y_train, K, seed)
    return X_train, Y_train, X_val, Y_val


def create_model(X_train, Y_train, X_val, Y_val):
    #somehow this comment (underneath) needs to be in the code otherwise it doesn't work..
    #h_nodes0 = {{choice([10, 20, 30, 40, 50])}} ; activation0 = {{choice(['tanh', 'sigmoid', 'relu', 'linear'])}} ; dropout0 = {{uniform(0, 1)}}
    input_size = 5 ; output_size = 1

    h_nodes = 3*[{{choice([10, 20, 30, 40, 50])}}]
    activation = 3*[{{choice(['tanh', 'sigmoid', 'relu', 'linear'])}}, 'linear']
    dropout = 3*[{{uniform(0, 1)}}]
    optimizer= {{choice(['sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'])}}
    activation_out = 'linear'    
    loss = 'mean_squared_error' 
    epochs = 15
    batch_size = 128
    
    nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, epochs, batch_size, optimizer)    

    cv_score = np.empty((5,2))
    for i in range(5):
        nn.MLP_model()
        result = nn.model.fit(X_train[i], Y_train[i], batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val[i], Y_val[i]))
        cv_score[i,0] = np.amin(result.history['val_loss']) ; cv_score[i,1] = np.amax(result.history['val_R_squared'])
    avg_cv_score = np.average(cv_score, axis=0) 
    print('Best validation acc of epoch:', avg_cv_score[1])
    return {'loss': avg_cv_score[0], 'status': STATUS_OK, 'model': nn.model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          functions=[cross_validation, R_squared],
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    X_train, Y_train, X_val, Y_val = data()
    print("Evalutation of best performing model:")
    score = np.empty((5,2))
    for i in range(5):
        score[i] = best_model.evaluate(X_val[i], Y_val[i])
    avg_score = np.average(score, axis=0)
    print(avg_score)
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
