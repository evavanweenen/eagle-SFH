#TODO: add bias
#TODO: See whether layers should be Dense or not fully connected
#TODO: Select best optimizer
#TODO: Select best activation function
from eagle.io import *
from eagle.plot import *

import numpy as np
from keras import backend as K
from keras.models import Sequential #Sequential NN
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, Adam, SGD
from sklearn.model_selection import KFold #cross-validation

#EAGLE settings
sim = 'RefL0050N0752' #simulation
cat = 'dusty-sdss' #catalogue to build model to
snapshot = 28 #redshift to use
L = 50 #box size


#ML settings manual
perc_train = .8
h_neurons = [30,30]
dropout = [.5, .5]
activation_h=['tanh', 'tanh']
activation_o='relu'
loss = 'mean_squared_error'
lr_rate = 0.001
epochs = 20
batch_size = 8
optimizer = 'Adam'

"""
#ML settings hyperas
perc_train = .8 #not with hyperas
h_neurons = [60,30]
dropout = [.06, .88]
activation_h = ['relu', 'relu']
activation_o='relu' #not with hyperas
loss = 'mse'
lr_rate = 0.001 #not with hyperas
epochs = 30
batch_size = 8
optimizer = 'Nadam'
"""

seed = 7
np.random.seed(seed) #fix seed for reproducibility

#data settings
dtype = ['<i8','<i8','<f8','<f8','<f8','<f8','<f8','<f8','<f8', '<f8']

xnames=['log dusty sdss u', 'log dusty sdss g', 'log dusty sdss r', 'log dusty sdss i', 'log dusty sdss z']; ynames=['log $M_{*}$']
xcols = ['dusty_sdss_u', 'dusty_sdss_g', 'dusty_sdss_r', 'dusty_sdss_i', 'dusty_sdss_z'] ; ycols = ['m_star']

def preprocess(sim, cat, xcols, ycols, dtype, perc_train, skip_header=15):
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
    

def MLP_model(activation_h, activation_o, dropout, lr_rate, loss, optimizer):
    """
    Build a sequential NN with input size 'in_size', output size 'out_size', 
    number of hidden layers 'len(neurons)-2' and number of neurons per layer i 'neurons[i]'
    """
  
    print("Building model..")
    model = Sequential()

    model.add(Dense(h_neurons[0], input_dim=input_size))
    model.add(Activation(activation_h[0]))
    model.add(Dropout(dropout[0]))

    for i in range(1, len(h_neurons)):
        model.add(Dense(h_neurons[i]))
        model.add(Activation(activation_h[i]))
        model.add(Dropout(dropout[i]))

    model.add(Dense(output_size))
    model.add(Activation(activation_o))
    
    #adam = Adam(lr=lr_rate)
    model.compile(loss=loss, optimizer=optimizer, metrics=[coeff_determination])
    return model
   
# read data
x, y, x_train, y_train, x_test, y_test, xscaler, yscaler = preprocess(sim, cat, xcols, ycols, dtype, perc_train=perc_train)

input_size = len(x_train[0])
output_size = len(y_train[0])
neurons = [input_size] + h_neurons + [output_size]
print("Neurons in the network: ", neurons)
   
#CREATE GIF
#make architecture of network
model = MLP_model(activation_h=activation_h, activation_o=activation_o, dropout=dropout, lr_rate=lr_rate, loss=loss, optimizer=optimizer)

#train network
Y_pred, mse, r2 = [], [], [] 
for i in range(epochs):
    result = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0, validation_data=(x_test, y_test))
    testscore = model.evaluate(x_test, y_test, verbose=0)
    y_pred = yscaler.inverse_transform(model.predict(x_test))
    plot_sed_mstar(x=xscaler.inverse_transform(x_test), y=yscaler.inverse_transform(y_test), dataset='test', sim=sim, cat=cat, snapshot=snapshot, xnames=xnames, ynames=ynames, predict=True, y_pred = y_pred, score = testscore, MLtype='Multi-layer perceptron', epochs=i, hiddenneurons=h_neurons, activation=activation_h, dropout=dropout, optimizer=optimizer, batchsize=batch_size)
    plot_prediction_test(y=yscaler.inverse_transform(y_test), y_pred=y_pred, sim=sim, cat=cat, snapshot=snapshot, score=testscore, MLtype='Multi-layer perceptron', epochs=epochs, hiddenneurons=h_neurons, activation=activation_h, dropout=dropout, optimizer=optimizer, batchsize=batch_size)
    Y_pred.append(y_pred) ; mse.append(testscore[0]) ; r2.append(testscore[1])

#plot gif
gif_sed_mstar(x=xscaler.inverse_transform(x_test), y=yscaler.inverse_transform(y_test), Y_pred=Y_pred, dataset='test', sim=sim, cat=cat, snapshot=snapshot, xnames=xnames, ynames=ynames, mse=mse, r2=r2, MLtype='Multi-layer perceptron', epochs=epochs, hiddenneurons=h_neurons, activation=activation_h, dropout=dropout, optimizer=optimizer, batchsize=batch_size)


#DO NOT CREATE GIF
#make architecture of network
model = MLP_model(activation_h=activation_h, activation_o=activation_o, dropout=dropout, lr_rate=lr_rate, loss=loss, optimizer=optimizer)

#train final model
result = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test))
testscore = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test) 

#postprocessing
x_test, y_test, y_pred = postprocess(x_test, y_test, y_pred, xscaler, yscaler)

#plotting
plot_learning_curve(result, epochs, 'test')

plot_sed_mstar(x_test, y_test, dataset='test', sim=sim, cat=cat, snapshot=snapshot, xnames=xnames, ynames=ynames, predict=True, y_pred = y_pred, score = testscore, MLtype='Multi-layer perceptron', epochs=epochs, hiddenneurons=h_neurons, activation=activation_h, dropout=dropout, optimizer=optimizer, batchsize=batch_size)

plot_prediction_test(y=y_test, y_pred=y_pred, sim=sim, cat=cat, snapshot=snapshot, score=testscore, MLtype='Multi-layer perceptron', epochs=epochs, hiddenneurons=h_neurons, activation=activation_h, dropout=dropout, optimizer=optimizer, batchsize=batch_size)
        

