from eagle.io import *
from eagle.plot import *

import numpy as np
from keras import backend as K
from keras.models import Sequential #Sequential NN
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, Adam, SGD
from sklearn.model_selection import KFold #cross-validation
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from tabulate import tabulate

#EAGLE settings
sim = 'RefL0050N0752' #simulation
cat = 'sdss' #catalogue to build model to
snapshot = 28 #redshift to use
L = 50 #box size

#optimize settings
Kcross = 5 

seed = 7
np.random.seed(seed) #fix seed for reproducibility

#data settings
dtype = ['<i8','<i8','<f8','<f8','<f8','<f8','<f8','<f8','<f8', '<f8']

xcols = ['sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z']
ycols = ['m_star']

dir =  '/disks/strw9/vanweenen/mrp2/plots/optimization/'

def preprocess(sim, cat, xcols, ycols, dtype, perc_train=.8, skip_header=15):
    #read data
    data = read_data(sim, cat, dtype=dtype, skip_header=skip_header)

    #select redshift
    data = select_redshift(data, snapshot)

    #divide data into x and y
    x, y = divide_input_output(data, xcols, ycols)
    
    #scale data to a logarithmic scale and then scale linearly to values between 0 and 1
    x, y = rescale_log(x,y)
    x, y, xscaler, yscaler = rescale_lin(x, y)

    #divide data into train and test set
    x_train, y_train, x_test, y_test = perm_train_test(x, y, len(data), perc_train=perc_train)

    print("Total size of data: %s; size of training set: %s ; size of test set: %s"%(len(x), len(x_train), len(x_test)))

    """
    #convert x and y back to a numpy ndarray
    data = data_ndarray(x,y)
    """   
    return x, y, x_train, y_train, x_test, y_test, xscaler, yscaler

def postprocess(x_test, y_test, y_pred, xscaler, yscaler):
    #return x_test, y_test and y_pred to their original scale    
    x_test = invscale_lin(x_test, xscaler)
    y_test = invscale_lin(y_test, yscaler)
    y_pred = invscale_lin(y_pred, yscaler)
    return x_test, y_test, y_pred

def coeff_determination(y_true, y_pred):
    """
    Determination coefficient metric
    Arguments
        y_true  array of true values of y
        y_pred  array of predicted values of y
    Returns
        R2  determination coefficient
    """
    SSE = K.sum(K.square(y_true - y_pred))
    TSS = K.sum(K.square(y_true - K.mean(y_true)))
    return 1-SSE/(TSS +K.epsilon())
    

def cross_validation_data(X, Y, seed, K=5):
    """
    Split the training set K times into a true training set and a validation set
    """
    valscores = []
    histories = [] 
    kf = KFold(n_splits=K, random_state=seed)
    X_train = [] ; Y_train = []; X_val = []; Y_val = []
    for idx_train, idx_val in kf.split(X):
        X_train.append(X[idx_train])
        Y_train.append(Y[idx_train])
        X_val.append(X[idx_val])
        Y_val.append(Y[idx_val])       
    return X_train, Y_train, X_val, Y_val

def data():

    return X_train[i], Y_train[i], X_val[i], Y_val[i]
        
def create_model(x_train, y_train, x_val, y_val):
    """
    Build a sequential NN with input size 'in_size', output size 'out_size', 
    number of hidden layers 'len(neurons)-2' and number of neurons per layer i 'neurons[i]'
    """
  
    print("Building model..")
    model = Sequential()
    
    model.add(Dense(30, input_dim=5))
    model.add(Dense({{choice([5, 10, 20, 30, 40, 50, 60])}}, input_dim=5))
    model.add(Activation('relu'))  
    #model.add(Activation({{choice['relu', 'tanh', 'sigmoid']}}))
    model.add(Dropout({{uniform(0,1)}}))
    
    if conditional({{choice(['two', 'three'])}}) == 'two':
        model.add(Dense({{choice([5, 10, 20, 30, 40, 50, 60])}}))
        model.add(Activation('relu'))
        #model.add(Activation({{choice['relu', 'tanh', 'sigmoid']}}))
        model.add(Dropout({{uniform(0,1)}}))

    if conditional({{choice(['two', 'three'])}}) == 'three':
        model.add(Dense({{choice([5, 10, 20, 30, 40, 50, 60])}}))
        model.add(Activation('relu'))
        #model.add(Activation({{choice['relu', 'tanh', 'sigmoid']}}))
        model.add(Dropout({{uniform(0,1)}}))
    
    model.add(Dense(1))
    model.add(Activation('relu'))
    
    model.compile(loss='mse', optimizer='adam', metrics=[coeff_determination])

    result = model.fit(x_train, y_train, batch_size=8, epochs=20, verbose=2, validation_data=(x_val, y_val))
    val_loss = np.amin(result.history['val_loss'])       
    return {'loss': val_loss, 'status': STATUS_OK, 'model': model} 
    

architect, best_runs, models, mses, r2s = [],[],[],[],[]
x, y, x_train, y_train, x_test, y_test, xscaler, yscaler = preprocess(sim, cat, xcols, ycols, dtype)
X_train, Y_train, X_val, Y_val = cross_validation_data(x_train, y_train, seed=seed) 


for i in range(Kcross):
    best_run, best_model = optim.minimize(model=create_model, data=data, algo=tpe.suggest, max_evals=10, trials=Trials(), rseed=seed)
    x_train, y_train, x_val, y_val = optimize_read_data()              
    print("Evaluation of best performing model:")
    mse, r2 = best_model.evaluate(x_val, y_val)
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    #save values    
    best_runs.append(best_run)
    architect.append(best_model)
    mses.append(mse)
    r2s.append(r2)
    best_model.save(dir+'model_hyp_%s.h5'%i)

#print(tabulate(np.array([range(10), mses, r2s, headers=['Model', 'Best Run MSE', 'r2']])))
