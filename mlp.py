from eagle.io import *
from eagle.plot import *
from eagle.nn import *

import numpy as np

seed = 7
np.random.seed(seed) #fix seed for reproducibility

#EAGLE settings
fol = ''
sim = 'RefL0100N1504' #simulation
cat = 'dusty-sdss' #catalogue to build model to
snap = 27 #redshift to use
#L = 100 #box size

#input output
inp = 'dusty-sdss'
outp = 'mstar'#'sfr'
xnames=['log dusty sdss ' + f for f in ['u','g','r','i','z']]
ynames=['log $M_{*}$']#['log SFR']
xcols = ['dusty_sdss_' + f for f in ['u','g','r','i','z']]
ycols = ['m_star']#['sfr']

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

def preprocess(io, dtype = ['<i8','<i8','<f8','<f8','<f8','<f8','<f8','<f8','<f8', '<f8'], skip_header=15):
    #read data     
    io.read_data(dtype=dtype, skip_header=skip_header)

    #divide data into x and y
    x, y = io.divide_input_output()
    
    #scale data to a logarithmic scale and then scale linearly to values between 0 and 1
    x = io.rescale_log(x)
    y = io.rescale_log(y)
    x, xscaler = io.rescale_lin(x)
    y, yscaler = io.rescale_lin(y)

    #divide data into train and test set
    x_train, y_train, x_test, y_test = io.perm_train_test(x, y)

    print("Total size of data: %s; size of training set: %s ; size of test set: %s"%(len(x), len(x_train), len(x_test)))
    return x, y, x_train, y_train, x_test, y_test, xscaler, yscaler

def postprocess(io, x_test, y_test, y_pred, xscaler, yscaler):
    #return x_test, y_test and y_pred to their original scale    
    x_test = io.invscale_lin(x_test, xscaler)
    y_test = io.invscale_lin(y_test, yscaler)
    y_pred = io.invscale_lin(y_pred, yscaler)
    return x_test, y_test, y_pred


#------------------------------------------DO NOT CREATE GIF--------------------------------------------------------------
#read data
io = IO(fol, sim, cat, snap, xcols, ycols, perc_train)
x, y, x_train, y_train, x_test, y_test, xscaler, yscaler = preprocess(io)

input_size = len(x_train[0])
output_size = len(y_train[0])
neurons = [input_size] + h_neurons + [output_size]
print("Neurons in the network: ", neurons)

#read hyperparameters and make architecture of the network
nn = NN(input_size, output_size, h_neurons, activation_h, dropout, activation_o, loss, lr_rate, epochs, batch_size, optimizer)
nn.MLP_model()

#train
result = nn.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test))
#evaluate
score = nn.model.evaluate(x_test, y_test, verbose=0)
#predict
y_pred = nn.model.predict(x_test) 
#postprocessing
x_test, y_test, y_pred = postprocess(io, x_test, y_test, y_pred, xscaler, yscaler)

#plot
plot = Plot(io, nn, xnames, ynames, MLtype='MLP', score=score)#[mse[19], r2[19]]

#plotting
plot.plot_learning_curve(result, nn.epochs)
plot.plot_input_output(x_test, y_test, y_pred)
plot.plot_true_predict(y_test, y_pred)

  
#------------------------------------------CREATE GIF--------------------------------------------------------------
#read data
io = IO(fol, sim, cat, snap, xcols, ycols, perc_train)
x, y, x_train, y_train, x_test, y_test, xscaler, yscaler = preprocess(io)

input_size = len(x_train[0])
output_size = len(y_train[0])
neurons = [input_size] + h_neurons + [output_size]
print("Neurons in the network: ", neurons)

#read hyperparameters and make architecture of the network
nn = NN(input_size, output_size, h_neurons, activation_h, dropout, activation_o, loss, lr_rate, epochs, batch_size, optimizer)
nn.MLP_model()


#train network
Y_pred, mse, r2 = [], [], [] 
for i in range(epochs):
    #train    
    result = nn.model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0, validation_data=(x_test, y_test))
    #evaluate    
    score = nn.model.evaluate(x_test, y_test, verbose=0)
    #predict
    y_pred = nn.model.predict(x_test)
    #postprocess    
    x_test_log, y_test_log, y_pred = postprocess(io, x_test, y_test, y_pred, xscaler, yscaler)
    #plot
    plot = Plot(io, nn, xnames, ynames, MLtype='MLP', score=score, epochs=i)
    plot.plot_input_output(x_test_log, y_test_log, y_pred)
    #plot.plot_true_predict(y_test_log, y_pred)
    #save values for gif    
    Y_pred.append(y_pred) ; mse.append(score[0]) ; r2.append(score[1])

#plot gif
plot.gif_input_output(x_test_log, y_test_log, Y_pred=Y_pred, mse=mse, r2=r2)


"""
#-------------------------------------------- Test model from hyperas -------------------------------------------------------
dir_hyp =  '/disks/strw9/vanweenen/mrp2/optimization/'
hyp = '20190114A'

model = load_model(dir_hyp+'model_hyp_'+hyp+'.h5', custom_objects={'coeff_determination':coeff_determination})

perc_train = .8
epochs = 20

#retrieve this info manually from out file
loss = 'mean_squared_error'
batch_size = 8
optimizer = 'RMSprop'

#read data
io = IO(fol, sim, cat, snap, xcols, ycols, perc_train)
x, y, x_train, y_train, x_test, y_test, xscaler, yscaler = preprocess(io)

input_size = len(x_train[0])
output_size = len(y_train[0])
neurons = [input_size] + h_neurons + [output_size]
print("Neurons in the network: ", neurons)

#evaluate
score = model.evaluate(x_test, y_test, verbose=0)
#predict
y_pred = model.predict(x_test)

#plotting
plot = Plot(io, nn, xnames, ynames, MLtype='MLPHYPERAS', score=score)
plot.plot_input_output(x_test, y_test, y_pred)
plot.plot_true_predict(y_test, y_pred)
"""
