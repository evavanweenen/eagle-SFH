import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class SDSS:
    def __init__(self, datacols, xcols, ycols, redshift):
        self.dir = '/disks/strw9/vanweenen/mrp2/data/'
        self.file = 'SDSS_DR9.hdf5'
        self.datacols = datacols 
        self.xcols = xcols 
        self.ycols = ycols 
        self.redshift = redshift

    def read_data(self):
        f = h5py.File(self.dir+self.file, 'r')
        self.hf_data = f.get('Data')
        self.data = np.empty((self.hf_data[self.datacols[0]].shape[0], len(self.datacols)))
        for i, col in enumerate(self.datacols):
            self.data[:,i] = self.hf_data.get(col)[()]
        self.data = to_structured_array(self.data, self.datacols, dtype = len(self.datacols)*['<f8'])

    def select_redshift(self, frac = 5e-8):
        self.data = self.data[np.where(self.data['Redshift'] > self.redshift - frac) and np.where(self.data['Redshift'] < self.redshift + frac)]

    def preprocess(self, frac = 5e-8):
        self.read_data()
        #select only galaxies of given redshift
        self.select_redshift(frac)
        #divide data into x and y
        x, y = divide_input_output(self)

        #scale only x data to a logarithmic scale 
        x = rescale_log(x)

        #remove infinite values caused by logarithmic scaling
        x, y = remove_inf(x, y)

        #scale data to standard scale with mean 0 and covariance 1
        x, self.xscaler = rescale_standard(x, EAGLE.xscaler)
        y, self.yscaler = rescale_standard(y, EAGLE.yscaler)
        return x, y

    def postprocess(self, x, y, y_pred):
        x = invscale(x, self.xscaler)
        y = invscale(y, self.yscaler)
        y_pred = invscale(y_pred, self.yscaler)
        return x, y, y_pred

class EAGLE:
    def __init__(self, fol, sim, cat, snap, xcols, ycols, perc_train, dtype=['<i8','<i8','<f8','<f8','<f8','<f8','<f8','<f8','<f8', '<f8'], skip_header=15):
        self.dir = '/disks/strw9/vanweenen/mrp2/data/'            
        self.fol = fol                  # location of file
        self.sim = sim                  # simulation
        self.cat = cat                  # fluxes catalogue
        self.snap = snap                # snapshot
        self.xcols = xcols              # column names of input (list)
        self.ycols = ycols              # column names of output (list)
        self.datacols = self.xcols + self.ycols
        self.perc_train = perc_train
        self.dtype = dtype
        self.skip_header = skip_header    
        self.path = self.dir + self.fol + self.sim + '-' + self.cat + '-snap' + str(self.snap) + '.csv'      

    def read_data(self):
        """
        Read csv file with data from eagle
        """
        print("Reading data with of simulation %s, catalogue %s"%(self.sim, self.cat))     
        self.data = np.genfromtxt(self.path, delimiter=',', names=True, dtype=self.dtype, skip_header=self.skip_header)
    
    def preprocess(self):
        #read data     
        self.read_data()

        #divide data into x and y
        x, y = divide_input_output(self)

        #scale data to a logarithmic scale and then scale to standard scale with mean 0 and covariance 1
        x = rescale_log(x)
        y = rescale_log(y)
        self.x, self.xscaler = rescale_standard(x)
        self.y, self.yscaler = rescale_standard(y)

        #divide data into train and test set
        x_train, y_train, x_test, y_test = perm_train_test(self)

        print("Total size of data: %s; size of training set: %s ; size of test set: %s"%(len(x), len(x_train), len(x_test)))
        return x_train, y_train, x_test, y_test

    def postprocess(self, x_test, y_test, y_pred):
        #return x_test, y_test and y_pred to their original scale  
        x_test = invscale(x_test, self.xscaler)
        y_test = invscale(y_test, self.yscaler)
        y_pred = invscale(y_pred, self.yscaler)
        return x_test, y_test, y_pred

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

def select_cols(self):
    self.data = self.data[self.datacols]

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

def rescale_log(a):
    """
    Scale the data to a logarithmic scale
    """
    return np.log10(a)

def rescale_lin(a, scaler=MinMaxScaler(feature_range=(0,1)):
    """
    Scale the data to a linear scale between 0 and 1
    """
    a = scaler.fit_transform(a)
    return a, scaler

def rescale_standard(a, scaler=StandardScaler()):
    """
    Scale the data to a standard scale with mean 0 and covariance 1
    """
    a = scaler.fit_transform(a)
    return a, scaler

def invscale(a, scaler):
    """
    Transform x and y back to their original values
    """    
    return scaler.inverse_transform(a)

def remove_inf(x, y):
    x_mask = ~np.isinf(x).any(axis=1)
    x = x[x_mask]
    y = y[x_mask]
    y_mask = ~np.isinf(y).any(axis=1)    
    x = x[y_mask]
    y = y[y_mask]
    return x, y

def perm_train_test(self):
    """
    Apply a random permutation to the data and put perc_train percent in the training set and the remainder in the test set
    """    
    print("Permuting data..")    
    #permute the data into the training and test set
    self.perm = np.random.choice([True, False], len(self.data), p=[self.perc_train, 1-self.perc_train])
    return self.x[self.perm,:], self.y[self.perm], self.x[np.invert(self.perm),:], self.y[np.invert(self.perm)]

def to_structured_array(a, acols, dtype):
    """
    Convert x and y back to a numpy ndarray
    """
    #convert array to structured array
    return np.core.records.fromarrays(a.transpose(), names=acols, formats=dtype)

def merge_structured_x_y(x, y):
    """
    Merge structured ararys x and y back together
    """   
    #merge x and y back together
    return np.lib.recfunctions.merge_arrays([x,y], flatten=True)
