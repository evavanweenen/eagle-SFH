import numpy as np
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, SGD
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


class NN():
    def __init__(self, input_size, output_size, h_neurons, activation_h, dropout, activation_o, loss, lr_rate, epochs, batch_size, optimizer):
        self.input_size = input_size
        self.output_size = output_size
        self.h_neurons = h_neurons
        self.activation_h = activation_h
        self.dropout = dropout
        self.activation_o = activation_o
        self.loss = loss
        self.lr_rate = lr_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
            
    def coeff_determination(self, y_true, y_pred):
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
    
    def MLP_model(self):
        """
        Build a sequential NN with input size 'in_size', output size 'out_size', 
        number of hidden layers 'len(neurons)-2' and number of neurons per layer i 'neurons[i]'
        """
        print("Building model..")
        # first hidden layer
        self.model = Sequential()
        self.model.add(Dense(self.h_neurons[0], input_dim=self.input_size, use_bias=True))
        self.model.add(Activation(self.activation_h[0]))
        self.model.add(Dropout(self.dropout[0]))
        # rest of hidden layers
        for i in range(1, len(self.h_neurons)):
            self.model.add(Dense(self.h_neurons[i], use_bias=True))
            self.model.add(Activation(self.activation_h[i]))
            self.model.add(Dropout(self.dropout[i]))
        #output layer
        self.model.add(Dense(self.output_size))
        self.model.add(Activation(self.activation_o))
        #compile model
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.coeff_determination])
        return self.model   

    def GridSearch(self):
        #TODO: checken of self.model returnen nodig is bij KerasClassifier
        #https://blogs.oracle.com/meena/simple-neural-network-model-using-keras-and-grid-search-hyperparameterstuning
        #TODO: finish grid search
        model = KerasClassifier(build_fn=self.MLP_model, verbose=0)
        param_grid = dict(h_neurons=self.h_neurons, dropout=self.dropout, activation_h=self.activation_h, activation_o=self.activation_o, batch_size=self.batch_size, optimizer=self.optimizer)        
        self.grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)

    def hyperas_create_model(self, X_train, Y_train, X_val, Y_val):
        self.MLP_model()

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
