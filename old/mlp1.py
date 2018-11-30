#TODO: add bias
#TODO: See whether layers should be Dense or not fully connected
#TODO: Select best optimizer
#TODO: Select best activation function

import numpy as np
from keras.models import Sequential #Sequential NN
from keras.layers import Dense, Dropout #Fully connected layers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score #cross-validation
from sklearn.model_selection import KFold #cross-validation
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline

#EAGLE settings
sim = 'RefL0050N0752' #simulation
cat = 'sdss' #catalogue to build model to
snapshot = 0 #redshift to use

#ML settings
perc_train = 0.8 #TODO
h_neurons = [500, 500] #TODO

batch_size = 128 #TODO
epochs = 20 #TODO
seed = 7
K = 5 #number of folds of cross-validation

np.random.seed(seed) #fix seed for reproducibility

dir = '/disks/strw9/vanweenen/mrp2/test/'
fname = sim+'-'+cat+'.csv'

def read_data(dir,fname):
    print("Reading data with of simulation %s, catalogue %s and snapshot %s"%(sim, cat, snapshot))
    dtype=['<i8','<i8','<f8','<f8','<f8','<f8','<f8','<f8','<f8', '<f8']
    return np.genfromtxt(dir+fname, delimiter=',', dtype=dtype, skip_header=15, names=True)

def preprocess(data):
    """
    Preprocess the data by selecting all galaxies from one redshift snapshot 's'
    Apply a random permutation to the data and put 'p' percent in the training set
    """    
    print("Preprocessing data..")    
    data = data[np.where(data['z'] == data['z'][snapshot])]
    x = data[['sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z']]
    y = data[['m_star']]#, 'sfr']]
    
    x = np.array(x.tolist()) #convert structured array to array
    y = np.array(y.tolist()) #convert structured array to array

    perm = np.random.choice([True, False], len(data), p=[perc_train, 1-perc_train])

    return x, y, x[perm,:], y[perm], x[np.invert(perm),:], y[np.invert(perm)]

def MLP_model():
    """
    Build a sequential NN with input size 'in_size', output size 'out_size', 
    number of hidden layers 'len(neurons)-2' and number of neurons per layer i 'neurons[i]'
    """
    print("Building model..")
    model = Sequential()
    # create model
    model.add(Dense(h_neurons[0], input_dim=5, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(h_neurons[1], activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    # compile model
    model.compile(loss='mse', optimizer='adam')
    return model

def evaluate(x_train, y_train):
    """
    Fit and evaluate model using K-fold cross-validation. 
    Return results. 
    """ 
    print("Evaluating model..")
    estimator = KerasRegressor(build_fn = MLP_model, epochs=epochs, batch_size=batch_size, verbose=True)
    kfold = KFold(n_splits=K, random_state=seed)
    return cross_val_score(estimator, x_train, y_train, cv=kfold)

def main(dir, fname):
    #read and preprocess data
    data = read_data(dir, fname)
    x_tot, y_tot, x_train, y_train, x_test, y_test = preprocess(data)  
    print("Total size of data: %s; size of training set: %s ; size of test set: %s"%(len(x_tot), len(x_train), len(x_test)))  

    #determine size of model
    input_size = len(x_train[0])
    output_size = len(y_train[0])
    neurons = [input_size] + h_neurons + [output_size]
    print("Neurons in the network: ", neurons)
    
    #make architecture of network
    model = MLP_model()
    print("Model ", model)

    #evaluate model
    results = evaluate(x_train, y_train)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

main(dir,fname)
    




