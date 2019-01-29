import numpy as np
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, SGD
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor


class NN:
    def __init__(self, input_size, output_size, h_nodes, activation, dropout, loss, lr_rate, epochs, batch_size, optimizer):
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
        self.lr_rate = lr_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        
 
    def MLP_model(self):
        """
        Build a sequential NN with input size 'in_size', output size 'out_size', 
        number of hidden layers 'len(neurons)-2' and number of neurons per layer i 'neurons[i]'
        """
        """
        h_nodes0 = self.h_nodes[0] ; activation0 = self.activation[0] ; dropout0 = self.dropout[0]
        h_nodes1 = self.h_nodes[1] ; activation1 = self.activation[1] ; dropout1 = self.dropout[1]
        h_nodes2 = self.h_nodes[2] ; activation2 = self.activation[2] ; dropout2 = self.dropout[2]
        activation_out = self.activation[3]
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
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[coeff_determination])

        return self.model

    def sklearn_wrapper(self, h_nodes0, activation0, h_nodes1, activation1, h_nodes2, activation2):
        self.h_nodes0 = h_nodes0 ; self.activation0 = activation0 #; self.dropout0 = dropout0
        self.h_nodes1 = h_nodes1 ; self.activation1 = activation1 #; self.dropout1 = dropout1
        self.h_nodes2 = h_nodes2 ; self.activation2 = activation2 #; self.dropout2 = dropout2    
        return self.MLP_model()

    def GridSearch(self, x, y, folds=5):
        #TODO: checken of self.model returnen nodig is bij KerasClassifier
        #https://blogs.oracle.com/meena/simple-neural-network-model-using-keras-and-grid-search-hyperparameterstuning
        #TODO: finish grid search
        model = KerasRegressor(build_fn=self.sklearn_wrapper, verbose=0)
        param_grid = dict(  h_nodes0 = self.h_nodes[0], activation0 = self.activation[0], \
                            h_nodes1 = self.h_nodes[1], activation1 = self.activation[1], \
                            h_nodes2 = self.h_nodes[2], activation2 = self.activation[2], )
        self.grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_'+self.loss, cv=folds, verbose=1, n_jobs=2)

        self.grid_result = self.grid.fit(x, y)

        print("Best: %f using %s" %(self.grid_result.best_score_, self.grid_result.best_params_))
        self.means = self.grid_result.cv_results_['mean_test_score']
        self.stds = self.grid_result.cv_results_['std_test_score']
        self.params = self.grid_result.cv_results_['params']
        for mean, std, param in zip(self.means, self.stds, self.params):
            print("%f (%f) with: %r" %(mean, std, param))

    def hyperas_create_model(self, X_train, Y_train, X_val, Y_val):
        self.MLP_model()


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



"""        
class OPTIM(NN):
    def __init__(self, input_arr, output_arr, h_neurons_arr, activation_h_arr, dropout_arr, activation_o_arr, loss_arr, lr_rate_arr, epochs_arr, batch_size_arr, optimizer_arr):
        self.input_arr = 
"""
"""
def OPTIM(NN):
    def __init__(self):

        
    def read_hyperparams(self):
        neurons = []
        activation = []
        self.dropout = []
        for layer in self.model.get_config()['layers']:
            if layer['class_name'] == 'Dense':
                neurons.append(layer['config']['units'])
            elif layer['class_name'] == 'Activation':
                activation.append(layer['config']['activation'])
            elif layer['class_name'] == 'Dropout':
                self.dropout.append(layer['config']['rate'])

        self.h_neurons = neurons[:-1]
        self.activation_h = activation[:-1]
        self.activation_o = activation[-1]
    
    def load_MLP_model(self, path, hyp):
        self.model = load_model(path+'model_hyp_'+hyp+'.h5', custom_objects={'coeff_determination':coeff_determination})
        read_hyperparams(self)
"""   
