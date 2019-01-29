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
snap = 27 #snapshot to use
redshift = 0.1

#input output
inp = 'dusty-sdss'
outp = 'mstar'#'sfr'
xnames=['$\log_{10}$ dusty sdss flux ' + f + ' (Jy) ' for f in ['u','g','r','i','z']]
ynames=['$\log_{10} M_{*} (M_{\odot})$']#['log SFR']

#eagle
eagle_xtype = 'flux'
eagle_xcols = ['dusty' + eagle_xtype + '_sdss_' + f for f in ['u','g','r','i','z']]
eagle_ycols = ['m_star']#['sfr']

#sdss
sdss_xtype = ''
sdss_xcols = ['flux_' + f + sdss_xtype for f in ['u', 'g', 'r', 'i', 'z']]
sdss_ycols = ['Mstellar_median']
sdss_datacols = sdss_xcols + sdss_ycols

#ML settings default (now for maximum of 3 hidden layers)
perc_train = .8
h_nodes = [30, 30, None]
dropout = [.5, .5, None]
activation = ['tanh', 'tanh', None, 'linear']
loss = 'mean_squared_error'
lr_rate = 0.001
epochs = 20
batch_size = 64#8
optimizer = 'Adam'


#------------------------------------------DO NOT CREATE GIF--------------------------------------------------------------
#read data
eagle = EAGLE(fol, sim, cat, snap, redshift, eagle_xcols, eagle_ycols, perc_train)
x_train, y_train, x_test, y_test = eagle.preprocess(eagle_xtype)

input_size = len(x_train[0])
output_size = len(y_train[0])
nodes = [input_size] + h_nodes + [output_size]
print("nodes in the network: ", nodes)

#read hyperparameters and make network
nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, lr_rate, epochs, batch_size, optimizer)
nn.MLP_model()

#train, evaluate, predict, postprocess
result = nn.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test)) #train
score = nn.model.evaluate(x_test, y_test, verbose=0) #evaluate
y_pred = nn.model.predict(x_test) #predict
x_test, y_test, y_pred = eagle.postprocess(x_test, y_test, y_pred) #postprocessing

#plot
plot = PLOT_NN(eagle, nn, xnames, ynames, MLtype='MLP', score=score)#[mse[19], r2[19]]
plot.plot_learning_curve(result, nn.epochs)
plot.plot_input_output(x_test, y_test, y_pred)
plot.plot_true_predict(y_test, y_pred)


#------------------------------------------Compare with SDSS--------------------------------------------------------------
#read data
sdss = SDSS(sdss_datacols, sdss_xcols, sdss_ycols, redshift = 0.1)
sdss.preprocess(eagle)

#plot data
plot_data = PLOT_DATA(xnames+ynames, sim=sim, cat=cat, snap=snap)
plot_data.hist_data(('eagle-preprocessed-minmax', 'sdss-preprocessed-minmax'), [np.hstack((eagle.x, eagle.y)), np.hstack((sdss.x, sdss.y))], xlim=[-1,2], ylim=[-2,1.5])

#evaluate, predict, postprocess
sdss_score = nn.model.evaluate(sdss.x, sdss.y, verbose=0) #evaluate
sdss.ypred = nn.model.predict(sdss.x) #predict
sdss.x, sdss.y, sdss.ypred = sdss.postprocess(eagle, sdss.x, sdss.y, sdss.ypred) #postprocessing

#plot
plot = PLOT_NN(eagle, nn, xnames, ynames, MLtype='MLPsdss', score=sdss_score)#[mse[19], r2[19]]
plot.plot_input_output(sdss.x, sdss.y, sdss.ypred)
plot.plot_true_predict(sdss.y, sdss.ypred)


#------------------------------------------CREATE GIF--------------------------------------------------------------
#read data
eagle = EAGLE(fol, sim, cat, snap, redshift, eagle_xcols, eagle_ycols, perc_train)
x_train, y_train, x_test, y_test = eagle.preprocess(eagle_xtype)

input_size = len(x_train[0])
output_size = len(y_train[0])
nodes = [input_size] + h_nodes + [output_size]
print("nodes in the network: ", nodes)

#read hyperparameters and make network
nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, lr_rate, epochs, batch_size, optimizer)
nn.MLP_model()

Y_pred, mse, r2 = [], [], [] 
for i in range(epochs):
    #train, evaluate, predict, postprocess
    result = nn.model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0, validation_data=(x_test, y_test)) #train 
    score = nn.model.evaluate(x_test, y_test, verbose=0) #evaluate
    y_pred = nn.model.predict(x_test) #predict
    x_test_log, y_test_log, y_pred = eagle.postprocess(x_test, y_test, y_pred) #postprocess
    #plotting
    plot = PLOT_NN(eagle, nn, xnames, ynames, MLtype='MLP', score=score, epochs=i)
    #plot.plot_input_output(x_test_log, y_test_log, y_pred)
    #plot.plot_true_predict(y_test_log, y_pred)
    #save values for gif    
    Y_pred.append(y_pred) ; mse.append(score[0]) ; r2.append(score[1])

#plot gif
plot.gif_input_output(x_test_log, y_test_log, Y_pred=Y_pred, mse=mse, r2=r2)


"""
#-------------------------------------------- Test model from hyperas -------------------------------------------------------
#Hyperas
perc_train = .8
h_nodes = 3*[{{choice([5, 10, 20, 30, 40, 50, 60])}}]
dropout = 3*[{{uniform(0,1)}}]
activation = 3*[{{choice(['relu', 'tanh', 'sigmoid', 'linear'])}}]+[{{choice(['relu', 'linear'])}}]
loss = 'mean_squared_error'
#lr_rate = 0.001
epochs = 20
batch_size = {{choice([8, 16, 32, 64])}}
optimizer = {{choice(['sgd', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'])}}



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
io = EAGLE(fol, sim, cat, snap, redshift, eagle_xcols, eagle_ycols, perc_train)
x, y, x_train, y_train, x_test, y_test, xscaler, yscaler = preprocess(io)

input_size = len(x_train[0])
output_size = len(y_train[0])
nodes = [input_size] + h_nodes + [output_size]
print("nodes in the network: ", nodes)

#evaluate
score = model.evaluate(x_test, y_test, verbose=0)
#predict
y_pred = model.predict(x_test)

#plotting
plot = Plot(io, nn, xnames, ynames, MLtype='MLPHYPERAS', score=score)
plot.plot_input_output(x_test, y_test, y_pred)
plot.plot_true_predict(y_test, y_pred)
"""
