import numpy as np
from sklearn.preprocessing import MinMaxScaler

class IO:
    def __init__(self, fol, sim, cat, snap, xcols, ycols, perc_train):
        self.dir = '/disks/strw9/vanweenen/mrp2/data/'            
        self.fol = fol                  # location of file
        self.sim = sim                  # simulation
        self.cat = cat                  # fluxes catalogue
        self.snap = snap                # snapshot
        self.xcols = xcols              # column names of input (list)
        self.ycols = ycols              # column names of output (list)
        self.perc_train = perc_train
        self.path = self.dir + self.fol + self.sim + '-' + self.cat + '-snap' + str(self.snap) + '.csv'  
         
    def read_data(self, **kwargs):
        """
        Read csv file with data from eagle
        """
        print("Reading data with of simulation %s, catalogue %s"%(self.sim, self.cat))     
        self.data = np.genfromtxt(self.path, delimiter=',', names=True, **kwargs)
            
    def divide_input_output(self):
        """
        Divide data into input and output
        """
        #divide data into input and output
        x = self.data[self.xcols]
        y = self.data[self.ycols]
        
        #convert structured array to array
        x = np.array(x.tolist())
        y = np.array(y.tolist())
        return x, y

    def rescale_log(self, a):
        """
        Scale the data to a logarithmic scale
        """
        return np.log10(a)

    def rescale_lin(self, a):
        """
        Scale the data to a linear scale between 0 and 1
        """
        scaler = MinMaxScaler(feature_range=(0,1))
        a = scaler.fit_transform(a)
        return a, scaler

    def invscale_lin(self, a, scaler):
        """
        Transform x and y back to their original values
        """    
        a = scaler.inverse_transform(a)
        return a

    def perm_train_test(self, x, y):
        """
        Apply a random permutation to the data and put perc_train percent in the training set and the remainder in the test set
        """    
        print("Permuting data..")    
        #permute the data into the training and test set
        self.perm = np.random.choice([True, False], len(self.data), p=[self.perc_train, 1-self.perc_train])
        return x[self.perm,:], y[self.perm], x[np.invert(self.perm),:], y[np.invert(self.perm)]
        
    def data_ndarray(self, x, y):
        """
        Convert x and y back to a numpy ndarray
        """
        #convert array to structured array
        x = np.core.records.fromarrays(x.transpose(), names=self.xcols, formats=5*['<f8'])
        y = np.core.records.fromarrays(y.transpose(), names=self.ycols, formats=['<f8'])
        
        #merge x and y back together
        self.data = np.lib.recfunctions.merge_arrays([x,y], flatten=True)

    def write_data(self, arr, **kwargs):
        """
        Write csv file with data from eagle
        Arguments:
            dir         - location of file
            sim         - simulation
            cat         - fluxes catalogue
        Returns:
            data (numpy ndarray)
        """
        print("Writing data with of simulation %s, catalogue %s"%(self.sim, self.cat))    
        path = dir + self.fol + self.sim+'-'+self.cat+'.csv'    
        return np.savetxt(self.path, arr, delimiter=',', **kwargs)
    

