import numpy as np
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, SGD
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


class NN:
    def __init__(self, input_size, output_size, h_nodes, activation, dropout, loss, epochs, batch_size, optimizer):
        self.input_size = input_size
        self.output_size = output_size
        self.h_nodes = h_nodes
        self.activation = activation
        self.dropout = dropout
        self.h_nodes0 = self.h_nodes[0] ; self.activation0 = self.activation[0] ; self.dropout0 = self.dropout[0]
        self.h_nodes1 = self.h_nodes[1] ; self.activation1 = self.activation[1] ; self.dropout1 = self.dropout[1]
        self.h_nodes2 = self.h_nodes[2] ; self.activation2 = self.activation[2] ; self.dropout2 = self.dropout[2]
        self.activation_out = self.activation[3]
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
 
    def MLP_model(self):
        """
        Build a sequential NN with input size 'in_size', output size 'out_size', 
        number of hidden layers 'len(neurons)-2' and number of neurons per layer i 'neurons[i]'
        """
        print("Building model..")
        self.model = Sequential()

        # first hidden layer (0)
        self.model.add(Dense(self.h_nodes0, input_dim=self.input_size, use_bias=True))
        self.model.add(Activation(self.activation0))
        self.model.add(Dropout(self.dropout0))

        # second hidden layer (1)
        if self.h_nodes1 != None:
            self.model.add(Dense(self.h_nodes1, use_bias=True))
            self.model.add(Activation(self.activation1))
            self.model.add(Dropout(self.dropout1))

        # third hidden layer (2)
        if self.h_nodes2 != None:
            self.model.add(Dense(self.h_nodes2, use_bias=True))
            self.model.add(Activation(self.activation2))
            self.model.add(Dropout(self.dropout2))

        #output layer
        self.model.add(Dense(self.output_size))
        self.model.add(Activation(self.activation_out))

        #compile model
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[R_squared])

        return self.model

    def GridSearch(self, x, y, folds=5):
        #https://blogs.oracle.com/meena/simple-neural-network-model-using-keras-and-grid-search-hyperparameterstuning
        def sklearn_wrapper(h_nodes0, h_nodes1, h_nodes2, activation0, activation1, activation2, dropout0, dropout1, dropout2, optimizer):#TODO
            self.h_nodes0 = h_nodes0 ; self.activation0 = activation0 ; self.dropout0 = dropout0 #TODO
            self.h_nodes1 = h_nodes1 ; self.activation1 = activation1 ; self.dropout1 = dropout1
            self.h_nodes2 = h_nodes2 ; self.activation2 = activation2 ; self.dropout2 = dropout2
            self.optimizer = optimizer
            return self.MLP_model()
        model = KerasRegressor(build_fn=sklearn_wrapper, verbose=0)
        param_grid = dict(  h_nodes0 = self.h_nodes[0], activation0 = self.activation[0], dropout0 = self.dropout[0], \
                            h_nodes1 = self.h_nodes[1], activation1 = self.activation[1], dropout1 = self.dropout[1], \
                            h_nodes2 = self.h_nodes[2], activation2 = self.activation[2], dropout2 = self.dropout[2], \
                            optimizer = self.optimizer)#TODO
        self.grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=('neg_'+self.loss, 'r2'), refit='neg_'+self.loss, cv=folds, verbose=1, n_jobs=1)

        self.grid_result = self.grid.fit(x, y)

        print("Best: %f using %s" %(self.grid_result.best_score_, self.grid_result.best_params_))

    def RandomSearch(self, x, y, n_iter=1000, folds=5):
        #TODO: work in progress
        def sklearn_wrapper(h_nodes0, activation0, dropout0, h_nodes1, activation1, dropout1, h_nodes2, activation2, dropout2):
            self.h_nodes0 = h_nodes0 ; self.activation0 = activation0 ; self.dropout0 = dropout0
            self.h_nodes1 = h_nodes1 ; self.activation1 = activation1 ; self.dropout1 = dropout1
            self.h_nodes2 = h_nodes2 ; self.activation2 = activation2 ; self.dropout2 = dropout2
            self.optimizer = optimizer
            return self.MLP_model()

        model = KerasRegressor(build=sklearn_wrapper, verbose=0)
        param_dist = dict(  h_nodes0 = self.h_nodes[0], activation0 = self.activation[0], dropout0 = self.dropout[0], \
                            h_nodes1 = self.h_nodes[1], activation1 = self.activation[1], dropout1 = self.dropout[1], \
                            h_nodes2 = self.h_nodes[2], activation2 = self.activation[2], dropout2 = self.dropout[2], \
                            optimizer = self.optimizer)
        self.grid = RandomizedSearchCV(estimator=model, param_distributions=param_dist, scoring='neg_'+self.loss, cv=folds, verbose=1, n_jobs=1)

        self.grid_result = self.grid.fit(x,y)        

def R_squared(y_true, y_pred):
    """
    Determination coefficient metric. Add K.epsilon()=1e-8 to avoid division by zero.
    Arguments
        y_true    - array of true values of y
        y_pred    - array of predicted values of y
    Returns
        R2        - determination coefficient
    """
    SSE = K.sum(K.square(y_true - y_pred))
    TSS = K.sum(K.square(y_true - K.mean(y_true)))
    return 1-SSE/(TSS+K.epsilon())

def adjusted_R_squared(R2, p, n):
    """
    Calculate (unbiased) determination coefficient adjusted for bias towards higher population size.
    Arguments
        R2          - (biased) determination coefficient
        p           - population size (number of input features)
        n           - sample size
    Returns
        Adjusted R2 - (unbiased) adjusted determination coefficient
    """
    return 1 - (1 - R2)*(n-1)/(n-p-1)
