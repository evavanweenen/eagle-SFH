#TODO: add bias
#TODO: See whether layers should be Dense or not fully connected
#TODO: Select best optimizer
#TODO: Select best activation function
from eagle.io import *
from eagle.plot import *

import numpy as np
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, Adam, SGD

#EAGLE settings
fol = ''
sim = 'RefL0100N1504' #simulation
cat = 'dusty-sdss-snap27' #catalogue to build model to
snapshot = 27 #redshift to use
L = 100 #box size

#input output
inp = 'dusty-sdss'
outp = 'mstar'
xnames=['log dusty sdss u', 'log dusty sdss g', 'log dusty sdss r', 'log dusty sdss i', 'log dusty sdss z']; ynames=['log $M_{*}$']
xcols = ['dusty_sdss_u', 'dusty_sdss_g', 'dusty_sdss_r', 'dusty_sdss_i', 'dusty_sdss_z'] ; ycols = ['m_star']


#ML settings default
perc_train = .8
h_neurons = [30,30]
dropout = [.5, .5]
activation_h=['tanh', 'tanh']
activation_o='relu'
loss = 'mean_squared_error'
lr_rate = 0.001
epochs = 20
batch_size = 64#8
optimizer = 'Adam'

"""
#ML settings hyperas 20190113
perc_train = .8 #not with hyperas
h_neurons = [40,40,40]
dropout = [0.14606156412564053, 0.5524073716526275, 0.42840543768112105]
activation_h = ['tanh', 'relu', 'tanh']
activation_o='relu' #not with hyperas
loss = 'mse'
#lr_rate = 0.001 #not with hyperas
epochs = 20
batch_size = 8
optimizer = 'Adamax'
"""

seed = 7
np.random.seed(seed) #fix seed for reproducibility

#data settings
dtype = ['<i8','<i8','<f8','<f8','<f8','<f8','<f8','<f8','<f8', '<f8']

def preprocess(fol, sim, cat, xcols, ycols, dtype, perc_train, skip_header=15):
    #read data
    data = read_data(fol, sim, cat, dtype=dtype, skip_header=skip_header)

    #select redshift
    #data = select_redshift(data, snapshot)

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
    Determination coefficient metric. Add K.epsilon()=1e-8 to avoid division by zero.
    Arguments
        y_true  array of true values of y
        y_pred  array of predicted values of y
    Returns
        R2  determination coefficient
    """
    SSE = K.sum(K.square(y_true - y_pred))
    TSS = K.sum(K.square(y_true - K.mean(y_true)))
    return 1-SSE/(TSS+K.epsilon())
    

def MLP_model(activation_h, activation_o, dropout, loss, optimizer):
    """
    Build a sequential NN with input size 'in_size', output size 'out_size', 
    number of hidden layers 'len(neurons)-2' and number of neurons per layer i 'neurons[i]'
    """
  
    print("Building model..")
    model = Sequential()

    model.add(Dense(h_neurons[0], input_dim=input_size, use_bias=True))
    if activation_h[0] == 'LeakyReLU':
        model.add(LeakyReLU())
    else:
        model.add(Activation(activation_h[0]))
    model.add(Dropout(dropout[0]))

    for i in range(1, len(h_neurons)):
        model.add(Dense(h_neurons[i], use_bias=True))
        if activation_h[i] == 'LeakyReLU':
            model.add(LeakyReLU())
        else:
            model.add(Activation(activation_h[i]))
        model.add(Dropout(dropout[i]))

    model.add(Dense(output_size))
    model.add(Activation(activation_o))
    
    #adam = Adam(lr=lr_rate)
    model.compile(loss=loss, optimizer=optimizer, metrics=[coeff_determination])
    return model

  
#------------------------------------------CREATE GIF--------------------------------------------------------------
# read data
x, y, x_train, y_train, x_test, y_test, xscaler, yscaler = preprocess(fol, sim, cat, xcols, ycols, dtype, perc_train=perc_train)

input_size = len(x_train[0])
output_size = len(y_train[0])
neurons = [input_size] + h_neurons + [output_size]
print("Neurons in the network: ", neurons)

#make architecture of network
model = MLP_model(activation_h, activation_o, dropout, loss, optimizer)

#train network
Y_pred, mse, r2 = [], [], [] 
for i in range(epochs):
    #train    
    result = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0, validation_data=(x_test, y_test))
    #evaluate    
    testscore = model.evaluate(x_test, y_test, verbose=0)
    #postprocess    
    y_pred = yscaler.inverse_transform(model.predict(x_test))
    #plot    
    plot_input_output(inp, outp, x=xscaler.inverse_transform(x_test), y=yscaler.inverse_transform(y_test), dataset='test', sim=sim, cat=cat, snapshot=snapshot, xnames=xnames, ynames=ynames, predict=True, y_pred = y_pred, score = testscore, MLtype='MLP', ep=i, hneurons=h_neurons, act=activation_h, drop=dropout, optim=optimizer, b=batch_size)
    plot_prediction_test(outp, y=yscaler.inverse_transform(y_test), y_pred=y_pred, sim=sim, cat=cat, snapshot=snapshot, score=testscore, MLtype='MLP', ep=epochs, hneurons=h_neurons, act=activation_h, drop=dropout, optim=optimizer, b=batch_size)
    Y_pred.append(y_pred) ; mse.append(testscore[0]) ; r2.append(testscore[1])

#plot gif
gif_sed_mstar(x=xscaler.inverse_transform(x_test), y=yscaler.inverse_transform(y_test), Y_pred=Y_pred, dataset='test', sim=sim, cat=cat, snapshot=snapshot, xnames=xnames, ynames=ynames, mse=mse, r2=r2, MLtype='MLP', ep=epochs, hneurons=h_neurons, act=activation_h, drop=dropout, optim=optimizer, b=batch_size)


#------------------------------------------DO NOT CREATE GIF--------------------------------------------------------------
# read data
x, y, x_train, y_train, x_test, y_test, xscaler, yscaler = preprocess(fol, sim, cat, xcols, ycols, dtype, perc_train=perc_train)

input_size = len(x_train[0])
output_size = len(y_train[0])
neurons = [input_size] + h_neurons + [output_size]
print("Neurons in the network: ", neurons)

#make architecture of network
model = MLP_model(activation_h, activation_o, dropout, loss, optimizer)

#train final model
result = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test))
testscore = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test) 

#postprocessing
x_test, y_test, y_pred = postprocess(x_test, y_test, y_pred, xscaler, yscaler)

#plotting
plot_learning_curve(result, epochs, 'test')

plot_input_output(inp, outp, x_test, y_test, dataset='test', sim=sim, cat=cat, snapshot=snapshot, xnames=xnames, ynames=ynames, predict=True, y_pred = y_pred, score = testscore, MLtype='MLP', ep=epochs, hneurons=h_neurons, act=activation_h, drop=dropout, optim=optimizer, b=batch_size)

plot_prediction_test(outp, y=y_test, y_pred=y_pred, sim=sim, cat=cat, snapshot=snapshot, score=testscore, MLtype='MLP', ep=epochs, hneurons=h_neurons, act=activation_h, drop=dropout, optim=optimizer, b=batch_size)


#-------------------------------------------- Test model from hyperas -------------------------------------------------------
dir_hyp =  '/disks/strw9/vanweenen/mrp2/optimization/'
hyp = '20190114A'

model = load_model(dir_hyp+'model_hyp_'+hyp+'.h5', custom_objects={'coeff_determination':coeff_determination})

perc_train = .8
epochs = 20

#load hyperparameters from model
neurons = []
activation = []
dropout = []
info = model.get_config()['layers']
for layer in info:
    if layer['class_name'] == 'Dense':
        neurons.append(layer['config']['units'])
    elif layer['class_name'] == 'Activation':
        activation.append(layer['config']['activation'])
    elif layer['class_name'] == 'LeakyReLU':
        activation.append(layer['class_name'])
    elif layer['class_name'] == 'Dropout':
        dropout.append(layer['config']['rate'])

h_neurons = neurons[:-1]
activation_h = activation[:-1]
activation_o = activation[-1]

#retrieve this info manually from out file
loss = 'mean_squared_error'
batch_size = 8
optimizer = 'RMSprop'

#read data
x, y, x_train, y_train, x_test, y_test, xscaler, yscaler = preprocess(fol, sim, cat, xcols, ycols, dtype, perc_train=perc_train)
input_size = len(x_train[0])
output_size = len(y_train[0])

#predict using trained model
testscore = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test)

#plotting
plot_input_output(inp, outp, x_test, y_test, dataset='test', sim=sim, cat=cat, snapshot=snapshot, xnames=xnames, ynames=ynames, predict=True, y_pred = y_pred, score = testscore, MLtype='MLPHYPERAS'+hyp, ep=epochs, hneurons=h_neurons, act=activation_h, drop=dropout, optim=optimizer, b=batch_size)

plot_prediction_test(outp, y=y_test, y_pred=y_pred, sim=sim, cat=cat, snapshot=snapshot, score=testscore, MLtype='MLPHYPERAS'+hyp, ep=epochs, hneurons=h_neurons, act=activation_h, drop=dropout, optim=optimizer, b=batch_size)

