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

dir =  '/disks/strw9/vanweenen/mrp2/optimization/'

def preprocess():
    sim = 'RefL0050N0752' #simulation
    cat = 'sdss' #catalogue to build model to
    snapshot = 28 #redshift to use

    #read data
    data = read_data(sim, cat, dtype=['<i8','<i8','<f8','<f8','<f8','<f8','<f8','<f8','<f8', '<f8'], skip_header=15)

    #select redshift
    data = select_redshift(data, snapshot)

    #divide data into x and y
    xcols = ['sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z']
    ycols = ['m_star']
    x, y = divide_input_output(data, xcols, ycols)
    
    #scale data to a logarithmic scale and then scale linearly to values between 0 and 1
    x, y = rescale_log(x,y)
    x, y, xscaler, yscaler = rescale_lin(x, y)

    #divide data into train and test set
    x_train, y_train, x_test, y_test = perm_train_test(x, y, len(data), perc_train=.8)

    print("Total size of data: %s; size of training set: %s ; size of test set: %s"%(len(x), len(x_train), len(x_test)))
    return x_train, y_train

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
    i = 0 ; Kcross = 5
    seed = 7 ; np.random.seed(seed)

    x_train, y_train = preprocess()
    X_train, Y_train, X_val, Y_val = cross_validation(x_train, y_train, Kcross, seed)
    
    x_train, y_train, x_val, y_val = X_train[i], Y_train[i], X_val[i], Y_val[i]
    return x_train, y_train, x_val, y_val

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
    
def create_model(x_train, y_train, x_val, y_val):
    """
    Build a sequential NN with input size 'in_size', output size 'out_size', 
    number of hidden layers 'len(neurons)-2' and number of neurons per layer i 'neurons[i]'
    """
  
    print("Building model..")
    model = Sequential()
    
    model.add(Dense({{choice([5, 10, 20, 30, 40, 50, 60])}}, input_dim=5))
    model.add({{choice([LeakyReLU(), Activation('relu'), Activation('tanh'), Activation('sigmoid')])}})
    model.add(Dropout({{uniform(0,1)}}))
    
    if conditional({{choice(['two', 'three'])}}) == 'two':
        model.add(Dense({{choice([5, 10, 20, 30, 40, 50, 60])}}))
        model.add({{choice([LeakyReLU(), Activation('relu'), Activation('tanh'), Activation('sigmoid')])}})
        model.add(Dropout({{uniform(0,1)}}))

    if conditional({{choice(['two', 'three'])}}) == 'three':
        model.add(Dense({{choice([5, 10, 20, 30, 40, 50, 60])}}))
        model.add({{choice([LeakyReLU(), Activation('relu'), Activation('tanh'), Activation('sigmoid')])}})
        model.add(Dropout({{uniform(0,1)}}))
    
    model.add(Dense(1))
    model.add(Activation('relu'))
    
    model.compile(loss={{choice(['mse', 'mae'])}}, optimizer={{choice(['sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'])}}, metrics=[coeff_determination])

    result = model.fit(x_train, y_train, batch_size={{choice([8,16,32])}}, epochs=30, verbose=0, validation_data=(x_val, y_val))
    val_loss = np.amin(result.history['val_loss'])       
    return {'loss': val_loss, 'status': STATUS_OK, 'model': model} 

    
architect, best_runs, models, mses, r2s = [],[],[],[],[]

i = 0
best_run, best_model = optim.minimize(model=create_model, data=data, functions=[preprocess, cross_validation,coeff_determination], algo=tpe.suggest, max_evals=5, trials=Trials(), rseed=7)
best_model.save(dir+'model_hyp_%s.h5'%i)

x_train, y_train, x_val, y_val = data()              
print("Evaluation of best performing model:")
mse, r2 = best_model.evaluate(x_val, y_val)
print(mse, r2)
print("Best performing model chosen hyper-parameters:")
print(best_run)

#save values    
best_runs.append(best_run)
architect.append(best_model)
mses.append(mse)
r2s.append(r2)


#print(tabulate(np.array([range(10), mses, r2s, headers=['Model', 'Best Run MSE', 'r2']])))
