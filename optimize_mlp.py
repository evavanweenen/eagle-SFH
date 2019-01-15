from eagle.io import *
from eagle.plot import *

import numpy as np
from keras import backend as K
from keras.models import Sequential #Sequential NN
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.optimizers import RMSprop, Adam, SGD
from sklearn.model_selection import KFold #cross-validation
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperas.utils import eval_hyperopt_space

from tabulate import tabulate

import datetime
import sys

dir =  '/disks/strw9/vanweenen/mrp2/optimization/'

name = '-1B'

date = datetime.date.today().strftime('%Y%m%d')
f = open(dir+'hyperas_'+date+name+'.out', 'w')
sys.stdout = f

def preprocess():
    fol = ''
    sim = 'RefL0100N1504' #simulation
    cat = 'dusty-sdss-snap27' #catalogue to build model to
    snapshot = 27

    #read data
    data = read_data(fol, sim, cat, dtype=['<i8','<i8','<f8','<f8','<f8','<f8','<f8','<f8','<f8', '<f8'], skip_header=15)

    #divide data into x and y
    xcols = ['dusty_sdss_u', 'dusty_sdss_g', 'dusty_sdss_r', 'dusty_sdss_i', 'dusty_sdss_z']
    ycols = ['m_star']
    x, y = divide_input_output(data, xcols, ycols)
    
    #scale data to a logarithmic scale and then scale linearly to values between 0 and 1
    x, y = rescale_log(x,y)
    x, y, xscaler, yscaler = rescale_lin(x, y)

    #divide data into train and test set
    x_train, y_train, x_test, y_test = perm_train_test(x, y, len(data), perc_train=.8)

    print("Total size of data: %s; size of training set: %s ; size of test set: %s"%(len(x), len(x_train), len(x_test)))
    return x_train, y_train #discard test data for optimizing hyperparameters

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
    Kcross = 5
    seed = 7 ; np.random.seed(seed)

    x_train, y_train = preprocess()
    X_train, Y_train, X_val, Y_val = cross_validation(x_train, y_train, Kcross, seed)

    return X_train, Y_train, X_val, Y_val

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
    
def create_model(X_train, Y_train, X_val, Y_val):
    """
    Build a sequential NN with input size 'in_size', output size 'out_size', 
    number of hidden layers 'len(neurons)-2' and number of neurons per layer i 'neurons[i]'
    """
  
    print("Building model..")
    crossval_loss = [] ; crossval_r2 = []
    for i in range(5):    
        model = Sequential()
        
        model.add(Dense({{choice([5, 10, 20, 30, 40, 50, 60])}}, input_dim=5, use_bias=True))
        model.add({{choice([LeakyReLU(), ELU(), Activation('relu'), Activation('tanh'), Activation('sigmoid')])}})
        model.add(Dropout({{uniform(0,1)}}))
        
        if conditional({{choice(['one', 'two'])}}) == 'two':
            model.add(Dense({{choice([5, 10, 20, 30, 40, 50, 60])}}, use_bias=True))
            model.add({{choice([LeakyReLU(), ELU(), Activation('relu'), Activation('tanh'), Activation('sigmoid')])}})
            model.add(Dropout({{uniform(0,1)}}))

        if conditional({{choice(['two', 'three'])}}) == 'three':
            model.add(Dense({{choice([5, 10, 20, 30, 40, 50, 60])}}, use_bias=True))
            model.add({{choice([LeakyReLU(), ELU(), Activation('relu'), Activation('tanh'), Activation('sigmoid')])}})
            model.add(Dropout({{uniform(0,1)}}))
        
        model.add(Dense(1))
        model.add(Activation('relu'))
        
        model.compile(loss={{choice(['mse', 'mae'])}}, optimizer={{choice(['sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'])}}, metrics=[coeff_determination])
        #model.compile(loss='mse', optimizer='Adam', metrics=[coeff_determination]) select this if you want fixed loss and optimizer
    
        result = model.fit(X_train[i], Y_train[i], batch_size={{choice([8,16,32])}}, epochs=20, verbose=0, validation_data=(X_val[i], Y_val[i]))
        #result = model.fit(X_train[i], Y_train[i], batch_size=64, epochs=20, verbose=0, validation_data=(X_val[i], Y_val[i])) select this if you want fixed batch size
        val_loss = np.amin(result.history['val_loss']) ; val_r2 = np.amax(result.history['val_coeff_determination'])
        crossval_loss.append(val_loss) ; crossval_r2.append(val_r2)
    avg_crossval_loss = np.average(crossval_loss) ; avg_crossval_r2 = np.average(crossval_r2)
    print(space)    
    return {'loss': avg_crossval_loss, 'status': STATUS_OK, 'model': model} 

    
architect, best_runs, models, mses, r2s = [],[],[],[],[]

best_run, best_model = optim.minimize(model=create_model, data=data, functions=[preprocess, cross_validation, coeff_determination], algo=tpe.suggest, max_evals=10, trials=Trials(), rseed=7, eval_space=True)
best_model.save(dir+'model_hyp_'+date+name+'.h5')

print("Best performing model chosen hyper-parameters:", best_run)

#real_param_values = eval_hyperopt_space(space, best_run)
#print(real_param_values)

#evaluate
X_train, Y_train, X_val, Y_val = data()              
print("Evaluation of best performing model:")
mse = [] ; r2 = []
for i in range(5):
    mse_i, r2_i = best_model.evaluate(X_val[i], Y_val[i])
    mse.append(mse_i) ; r2.append(r2_i)
mse = np.average(mse) ; r2 = np.average(r2)
print(mse, r2)

#save values    
best_runs.append(best_run)
architect.append(best_model)
mses.append(mse)
r2s.append(r2)

f.close()

#print(tabulate(np.array([range(10), mses, r2s, headers=['Model', 'Best Run MSE', 'r2']])))k
