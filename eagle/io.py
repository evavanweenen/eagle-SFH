from .calc import *

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, append_fields

import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class SDSS:
    def __init__(self, fluxes, xcols, ycols, xtype, redshift):
        self.dir = '/disks/strw9/vanweenen/mrp2/data/'
        self.file = 'SDSS_DR9.hdf5'
        self.xtype = xtype
        self.xcols = xcols 
        self.ycols = ycols
        self.datacols = [xtype + '_' + f for f in fluxes] + ycols + ['Redshift']
        self.redshift = redshift

    def read_data(self):
        f = h5py.File(self.dir+self.file, 'r')
        self.hf_data = f.get('Data')
        self.data = np.empty((self.hf_data[self.datacols[0]].shape[0], len(self.datacols)))
        for i, col in enumerate(self.datacols):
            self.data[:,i] = self.hf_data.get(col)[()]
        self.data = to_structured_array(self.data, self.datacols, dtype = len(self.datacols)*['<f8'])

    def select_redshift(self, frac = 5e-3):
        self.data = self.data[np.where(self.data['Redshift'] > self.redshift*(1 - frac))]
        self.data = self.data[np.where(self.data['Redshift'] < self.redshift*(1 + frac))]

    def preprocess(self, eagle, colors, frac = 5e-3, scaling=True):
        self.read_data()

        #select only galaxies of given redshift
        self.select_redshift(frac)
        
        #to xtype
        to_xtype(self, self.xtype)

        #add colors
        add_colors(self, self.xtype+'_', colors)

        #select only columns necessary
        self.datacols = self.xcols + self.ycols
        select_cols(self)

        #divide data into x and y
        x, y = divide_input_output(self)

        #remove galaxies with zero y
        x, y = remove_zero(x, y)

        #remove infinite values caused by logarithmic scaling of x
        self.x, self.y = remove_inf(x, y)

        if scaling:
            #scale data to standard scale with mean 0 and covariance 1
            self.y = self.y.reshape(-1,1)
            self.x = eagle.xscaler.transform(self.x)
            self.y = eagle.yscaler.transform(self.y)

    def postprocess(self, eagle, x, y, y_pred):
        x = eagle.xscaler.inverse_transform(x)
        y = eagle.yscaler.inverse_transform(y)
        y_pred = eagle.yscaler.inverse_transform(y_pred)
        return x, y, y_pred

class EAGLE:
    def __init__(self, fol, sim, cat, dust, snap, redshift, seed=7):
        self.dir = '/disks/strw9/vanweenen/mrp2/data/'            
        self.fol = fol                      # location of file
        self.sim = sim                      # EAGLE simulation (e.g. 'RefL0100N1504')
        self.cat = cat                      # fluxes catalogue (e.g. 'sdss')
        self.dust = dust                    # 'dusty' or ''
        self.snap = snap                    # snapshot
        self.redshift = redshift            # redshift of snapshot
        self.path = self.dir + self.fol + self.sim + '-' + self.dust + self.cat + '-snap' + str(self.snap) + '.csv'
        self.seed = seed                    # random seed
        np.random.seed(self.seed)

    def read_data(self, dtype=2*['<i8']+['<f8']+['<i8']+12*['<f8'], skip_header=21):
        """
        Read csv file with data from eagle
        """
        print("Reading data with of simulation %s, catalogue %s"%(self.sim, self.dust+self.cat))
        self.data = np.genfromtxt(self.path, delimiter=',', names=True, dtype=dtype, skip_header=skip_header)
    
    def preprocess(self, colors, xcols, ycols, xtype='flux', equal_bins=False, bins=10, count=100, random_sample=False, N=900, scaling=True, perc_train=.8):
        """
            colors          - (tuple) colors to add to data (e.g. ('ug', 'gr', 'ri', 'iz'))
            xcols           - (list) column names of input
            ycols           - (list) column names of output
            xtype           - (string) input type that you read from file ('flux' or (absolute!) 'magnitude')
            scaling         - (boolean) whether to apply minmax or standardscaling
            equal_bins      - (boolean) whether to apply uniform mass sampling
            bins            - (int) the number of bins for the uniform mass sampling
            count           - (int) the number of galaxies per bin
            random_sample   - (boolean) whether to take a random sample
            N               - (int) the size of the random sample
            perc_train      - (float) fraction of training data
        """
        self.xtype = xtype
        self.xcols = xcols
        self.ycols = ycols
        self.datacols = self.xcols + self.ycols + ['subgroup']   # data columns

        #read data
        self.read_data()

        #convert fluxes to magnitudes if input is fluxes
        to_xtype(self, self.dust+self.xtype)

        #add colors data
        if len(colors) != 0:
            add_colors(self, self.dust+self.xtype+'_'+self.cat+'_', colors)

        #select only columns necessary
        select_cols(self)

        #rescale stellar mass to logarithm of stellar mass
        rescale_log(self)

        #divide data into x and y
        self.x, self.y = divide_input_output(self)

        #central satellite array
        self.cs = (self.data['subgroup'] == 0)

        #uniform mass distribution
        if equal_bins:
            equal_bin_distr(self, bins, count)

        if random_sample:
            random_sample(self, N)

        if scaling:
            self.y = self.y.reshape(-1,1)
            self.xscaler = MinMaxScaler(feature_range=(-1,1)).fit(self.x)#StandardScaler())#
            self.yscaler = MinMaxScaler(feature_range=(-1,1)).fit(self.y)#StandardScaler())#
            self.x = self.xscaler.transform(self.x)
            self.y = self.yscaler.transform(self.y)

        #divide data into train and test set
        x_train, x_test, y_train, y_test, self.cs_train, self.cs_test = train_test_split(self.x, self.y, self.cs, train_size=perc_train, random_state=self.seed, shuffle=True)

        print("Total size of data: %s; size of training set: %s ; size of test set: %s"%(len(self.x), len(x_train), len(x_test)))
        return x_train, y_train, x_test, y_test

    def postprocess(self, x_test, y_test, y_pred):
        #return x_test, y_test and y_pred to their original scale  
        x_test = self.xscaler.inverse_transform(x_test)
        y_test = self.yscaler.inverse_transform(y_test)
        y_pred = self.yscaler.inverse_transform(y_pred)
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
        print("Writing data with of simulation %s, catalogue %s"%(self.sim, self.dust+self.cat))
        path = dir + self.fol + self.sim+'-'+self.dust+self.cat+'.csv'
        return np.savetxt(self.path, arr, delimiter=',', **kwargs)

def to_xtype(self, pre):
    """
    If input are fluxes, convert fluxes to (absolute!) magnitudes, using ealge.calc functions.
    """
    if self.xtype == 'flux':
        for col in self.data.dtype.names:
            if col.startswith(pre):
                self.data[col] = app_to_abs_mag(flux_to_magAB(self.data[col]), luminosity_distance(self.redshift))

def add_colors(self, pre, colors = ('ug', 'gr', 'ri', 'iz')):
    """
    Add colors to data, calculated from magnitudes or from fluxes
    """
    colors = [list(c) for c in colors]
    for c in colors:
        self.data = append_fields(self.data, pre+c[0]+c[1], self.data[pre+c[0]] - self.data[pre+c[1]], usemask=False)

def select_cols(self):
    self.data = self.data[self.datacols]

def divide_input_output(self):
    """
    Divide data into input and output and convert to unstructured array
    """
    #divide data into input and output
    x = structured_to_unstructured(self.data[self.xcols])
    y = structured_to_unstructured(self.data[self.ycols])
    return x, y

def random_sample(self, N):
    """
    Take a random sample of size N from the data
    """
    perm = np.random.permutation(self.data.shape[0])[:N] #take first N numbers of a random permutation
    self.x = self.x[perm]
    self.y = self.y[perm]
    self.cs = self.cs[perm]

def equal_bin_distr(self, bins=10, count=100):
    """
    Uniform mass sampling
    """
    self.y = self.y.reshape(-1,1)
    hist, self.edges = np.histogram(self.y, bins=bins)

    #array with lower edges and upper edges of bins that are above the minimal count
    lower_edges = self.edges[np.insert(hist > count, len(hist), False)]
    upper_edges = self.edges[np.insert(hist > count, 0, False)]
    hist = hist[hist > count]

    #create new x and y data consisting of an equal number of galaxies per mass bin
    #for each bin, randomly shuffle the data in the bin, and then keep the first count galaxies
    x_equal = np.empty((count*len(hist), self.x.shape[1]))
    y_equal = np.empty((count*len(hist), self.y.shape[1]))
    cs_equal = np.zeros((count*len(hist)), dtype=bool)
    for i in range(len(lower_edges)):
        perm = np.random.permutation(hist[i])[:count]
        mask = (self.y >= lower_edges[i]) & (self.y < upper_edges[i])
        mask = mask.reshape(len(mask),)
        x_equal[i*count:(i+1)*count] = self.x[mask][perm]
        y_equal[i*count:(i+1)*count] = self.y[mask][perm]
        cs_equal[i*count:(i+1)*count] = self.cs[mask][perm]
    self.x = x_equal ; self.y = y_equal ; self.cs = cs_equal

def rescale_log(self):
    """
    Scale the data to a logarithmic scale
    """
    for col in self.ycols:
        if col == 'm_star':
            self.data['m_star'] = np.log10(self.data['m_star'])

def remove_zero(x, y):
    """
    Remove galaxies with zero mass    
    """
    x = x[y != 0.]
    y = y[y != 0.]
    return x, y

def remove_inf(x, y):
    """
    Remove galaxies with infinite values caused by 0's in logarithmic scaling
    """
    y = y[~np.isinf(x).any(axis=1)]
    x = x[~np.isinf(x).any(axis=1)]
    return x, y

def to_structured_array(a, acols, dtype):
    """
    Convert x and y back to a numpy ndarray
    """
    #convert array to structured array
    return np.core.records.fromarrays(a.transpose(), names=acols, formats=dtype)

def merge_x_y(x,y):
    return np.hstack((x, y))

def merge_structured_x_y(x, y):
    """
    Merge structured ararys x and y back together
    """   
    #merge x and y back together
    return np.lib.recfunctions.merge_arrays([x,y], flatten=True)
