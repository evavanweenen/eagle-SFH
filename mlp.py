from eagle.io import *
from eagle.plot import *
from eagle.nn import *

import numpy as np
import tensorflow as tf

seed = 7
np.random.seed(seed) #fix seed for reproducibility
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

#sdss
sdss_xtype = 'flux'
sdss_xcols = [sdss_xtype + '_' + f for f in xcols]
sdss_ycols = ['Mstellar_median']#['SFR_median']#

#input output plot
xnames=[dust + ' ' + cat + ' ' + i for i in list(fluxes) + [c[0] + '-' + c[1] for c in colors]]
ynames=['$\log_{10} M_{*} (M_{\odot})$']#['$\log_{10}$ SFR $(M_{\odot} yr^{-1})$']#

cs = False ; cs_mask = None #central satellite analysis (only change cs)
equal_bins = True #uniform mass distribution


#ML settings GridSearch
perc_train = .8 #todo: this is not used anywhere, give it to io
h_nodes = [20, 50, 50]
dropout = [0., 0., 0.]
activation = ['tanh', 'linear', 'tanh', 'linear']
loss = 'mean_squared_error'
epochs = 15
batch_size = 128
optimizer = 'Adamax'
"""
#Hyperas
perc_train = .8 #todo: this is not used anywhere, give it to io
h_nodes = [30, 50, 20]
dropout = [0.43, 0.697, 0.515]
activation = ['linear', 'tanh', 'linear', 'linear']
loss = 'mean_squared_error'
epochs = 15
batch_size = 128
optimizer = 'Adam'
"""
def add_scores(score, x, y, y_pred):
    y.reshape(len(y),) ; y_pred.reshape(len(y_pred),)
    score[1] = r2_score(y_test, y_pred)
    score += [adjusted_R_squared(score[1], x.shape[1], len(y)), np.mean(y - y_pred), np.var(y - y_pred)]
    return score

#------------------------------------------DO NOT CREATE GIF--------------------------------------------------------------
#read data
eagle = EAGLE(fol, sim, cat, dust, snap, redshift)
x_train, y_train, x_test, y_test = eagle.preprocess(colors, eagle_xcols, eagle_ycols, eagle_xtype, equal_bins)
"""
if equal_bins:
    plot_data = PLOT_DATA(xnames+ynames, sim=sim, cat=cat, snap=snap)
    plot_data.hist_data(['eagle-preprocessed-minmax-equalbins'], [np.hstack((eagle.x, eagle.y))], eagle.yscaler.transform(eagle.edges.reshape(-1,1)).T[0][:-1], xlim=[-1,1], ylim=[-1,1])
"""

input_size = len(x_train[0])
output_size = len(y_train[0])
nodes = [input_size] + h_nodes + [output_size]
print("nodes in the network: ", nodes)

#read hyperparameters and make network
nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, epochs, batch_size, optimizer)
nn.MLP_model()

#train, evaluate, predict, postprocess
result = nn.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test)) #train
score = nn.model.evaluate(x_test, y_test, verbose=0) #evaluate
y_pred = nn.model.predict(x_test) #predict
old_r2 = score[1]
score = add_scores(score, x_test, y_test, y_pred) #add scores
print("Errors (EAGLE): MSE = %.3e ; R^2 = %.2f ; adjusted R^2 = %.2f ; error_mean = %.2e ; error_sigma = %.2e"%tuple(score))

#central satellites analysis
if cs == True:
    cs_mask = eagle.cs_test
    score_c = nn.model.evaluate(x_test[cs_mask], y_test[cs_mask], verbose=0)
    score_s = nn.model.evaluate(x_test[~cs_mask], y_test[~cs_mask], verbose=0)
    score_c = add_scores(score_c, x_test[cs_mask], y_test[cs_mask], y_pred[cs_mask])
    score_s = add_scores(score_s, x_test[~cs_mask], y_test[~cs_mask], y_pred[~cs_mask])
    print("Errors (EAGLE) centrals: MSE = %.3e ; R^2 = %.2g ; adjusted R^2 = %.2g ; error_mean = %.2e ; error_sigma = %.2e"%tuple(score_c))
    print("Errors (EAGLE) satellites: MSE = %.3e ; R^2 = %.2g ; adjusted R^2 = %.2g ; error_mean = %.2e ; error_sigma = %.2e"%tuple(score_s))

x_test, y_test, y_pred = eagle.postprocess(x_test, y_test, y_pred) #postprocessing

#plot
plot = PLOT_NN(eagle, nn, xnames, ynames, MLtype='MLP', score=score)#[mse[19], r2[19]]

plot.plot_learning_curve(result, nn.epochs)

plot.plot_input_output(x_test, y_test, y_pred, 'scatter')
#plot.plot_input_output(x_test, y_test, y_pred, 'contour')

plot.plot_true_predict(y_test, y_pred, 'scatter', cs_mask)
#plot.plot_true_predict(y_test, y_pred, 'hexbin', cs_mask)
#plot.plot_true_predict(y_test, y_pred, 'contour', cs_mask)
#plot.plot_true_predict(y_test, y_pred, 'contourf', cs_mask)

#plot.plot_output_error(y_test, y_pred, 'scatter', cs_mask)
plot.plot_output_error(y_test, y_pred, 'hexbin', cs_mask)
#plot.plot_output_error(y_test, y_pred, 'contour', cs_mask)

#------------------------------------------Compare with SDSS--------------------------------------------------------------
#read data
sdss = SDSS(fluxes, sdss_xcols, sdss_ycols, sdss_xtype, redshift = 0.1)
sdss.preprocess(eagle, colors)

#plot data
#plot_data = PLOT_DATA(xnames+ynames, sim=sim, cat=cat, snap=snap)
#plot_data.hist_data(('eagle-preprocessed-minmax', 'sdss-preprocessed-minmax'), [np.hstack((eagle.x, eagle.y)), np.hstack((sdss.x, sdss.y))], xlim=[-1,2], ylim=[-2,1.5])

#evaluate, predict, postprocess
sdss_score = nn.model.evaluate(sdss.x, sdss.y, verbose=0) #evaluate
sdss.ypred = nn.model.predict(sdss.x) #predict
sdss_score = add_scores(sdss_score, sdss.x, sdss.y, sdss.ypred)
print("Errors (SDSS): MSE = %.3e ; R^2 = %.2g ; adjusted R^2 = %.2g ; error_mean = %.2e ; error_sigma = %.2e"%tuple(sdss_score))

sdss.x, sdss.y, sdss.ypred = sdss.postprocess(eagle, sdss.x, sdss.y, sdss.ypred) #postprocessing

#plot
plot = PLOT_NN(eagle, nn, xnames, ynames, MLtype='MLPsdss', score=sdss_score)#[mse[19], r2[19]]
plot.plot_input_output(sdss.x, sdss.y, sdss.ypred)
plot.plot_true_predict(sdss.y, sdss.ypred, 'scatter')
#plot.plot_true_predict(sdss.y, sdss.ypred, 'hexbin', bins=70)
#plot.plot_output_error(sdss.y, sdss.ypred, 'scatter')
plot.plot_output_error(sdss.y, sdss.ypred, 'hexbin', bins=80, ylim=(-0.8, .2))


"""
#------------------------------------------CREATE GIF--------------------------------------------------------------
#read data
eagle = EAGLE(fol, sim, cat, snap, redshift, eagle_xcols, eagle_ycols, perc_train)
x_train, y_train, x_test, y_test = eagle.preprocess(eagle_xtype)

input_size = len(x_train[0])
output_size = len(y_train[0])
nodes = [input_size] + h_nodes + [output_size]
print("nodes in the network: ", nodes)

#read hyperparameters and make network
nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, epochs, batch_size, optimizer)
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

