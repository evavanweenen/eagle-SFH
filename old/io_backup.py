from .calc import *

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, append_fields

import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class SDSS:
    def __init__(self, datacols, xcols, ycols, redshift):
        self.dir = '/disks/strw9/vanweenen/mrp2/data/'
        self.file = 'SDSS_DR9.hdf5'
        self.datacols = datacols +['Redshift']
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

    def select_redshift(self, frac = 5e-3):
        self.data = self.data[np.where(self.data['Redshift'] > self.redshift*(1 - frac))]
        self.data = self.data[np.where(self.data['Redshift'] < self.redshift*(1 + frac))]

    def preprocess(self, eagle, frac = 5e-3, scaling=True):        
        self.read_data()

        #select only galaxies of given redshift
        self.select_redshift(frac)
        
        #select only columns necessary
        self.datacols = self.xcols + self.ycols
        select_cols(self)

        #divide data into x and y
        x, y = divide_input_output(self)

        #remove galaxies with zero mass or zero flux
        x, y = remove_zero(x, y)

        #scale only x data to a logarithmic scale 
        x = rescale_log(x)

        #remove infinite values caused by logarithmic scaling
        self.x, self.y = remove_inf(x, y)
        
        if scaling:
            #scale data to standard scale with mean 0 and covariance 1
            self.x = rescale(self.x, eagle.xscaler)
            self.y = rescale(self.y, eagle.yscaler)

    def postprocess(self, eagle, x, y, y_pred):
        x = invscale(x, eagle.xscaler)
        y = invscale(y, eagle.yscaler)
        y_pred = invscale(y_pred, eagle.yscaler)
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
    
    def preprocess(self, fluxes, colors, xcols, ycols, xtype='flux', scaling=True, perc_train=.8):   
        self.xtype = xtype                                       # name of input that you read from file 
        self.fluxcols = [self.dust + inXtype + '_' + self.cat + '_' + f for f in fluxes]
        self.colorcols = [self.dust + inXtype + '_' + self.cat + '_' + c for c in colors]     
        self.xcols = xcols                                       # column names of input (list)
        self.ycols = ycols                                       # column names of output (list)     
        self.datacols = self.xcols + self.ycols + ['subgroup']   # data columns
        
        #convert magnitudes to fluxes or vice versa if inXtype and toXtype differ
        to_xtype(self)

        #add colors data
        add_colors(self, colors)



        #select only columns necessary
        select_cols(self)

        #divide data into x and y
        self.x, self.y = divide_input_output(self)

        #scale data to a logarithmic scale
        rescale_log(self)

        if scaling:
            #scale to standard scale with mean 0 and covariance 1
            self.xscaler = fitscale(self.x, scaler=MinMaxScaler(feature_range=(-1,1)))#StandardScaler())#
            self.yscaler = fitscale(self.y, scaler=MinMaxScaler(feature_range=(-1,1)))#StandardScaler())#
            self.x = rescale(self.x, self.xscaler)
            self.y = rescale(self.y, self.yscaler)

        #central satellite array
        self.cs = (self.data['subgroup'] == 0)

        #divide data into train and test set
        x_train, x_test, y_train, y_test, self.cs_train, self.cs_test = train_test_split(self.x, self.y, self.cs, random_state=self.seed, shuffle=True)
        
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
        print("Writing data with of simulation %s, catalogue %s"%(self.sim, self.dust+self.cat))    
        path = dir + self.fol + self.sim+'-'+self.dust+self.cat+'.csv'    
        return np.savetxt(self.path, arr, delimiter=',', **kwargs)

def to_xtype(self):
    """
    If inXtype and toXtype are different, convert inXtype to toXtype.
    Example: if inXtype = 'flux' and toXtype = 'magnitude', convert (only!) fluxes to magnitudes, using ealge.calc functions.
    """
    if self.inXtype == 'magnitude' and self.toXtype == 'flux':
    # convert data columns with absolute magnitudes to fluxes        
        for col in self.data.dtype.names:
            if col.startswith(self.dust + self.inXtype):       
                self.data[col] = magAB_to_flux(abs_to_app_mag(self.data[col], luminosity_distance(self.redshift)))
    
    elif self.inXtype == 'flux' and self.toXtype == 'magnitude':
    # convert data columns with fluxes to absolute magnitudes
        for col in self.data.dtype.names:
            if col.startswith(self.dust + self.inXtype):
                self.data[col] = app_to_abs_mag(flux_to_magAB(self.data[col]), luminosity_distance(self.redshift))

def add_colors(self, colors = ('ug', 'gr', 'ri', 'iz')):
    """
    Add colors to data, calculated from magnitudes or from fluxes
    """
    pre = self.dust + self.inXtype + '_' + self.cat + '_'   
    colors = [list(c) for c in colors]    
    for c in colors:
        if self.toXtype == 'flux':
            append_fields(self.data, pre+c[0]+c[1], self.data[pre+c[0]] / self.data[pre+c[1]], usemask=False)
        elif self.toXtype == 'magnitude':
            append_fields(self.data, pre+c[0]+c[1], self.data[pre+c[0]] - self.data[pre+c[1]], usemask=False)

def select_cols(self):
    self.data = self.data[self.datacols]

def divide_input_output(self):
    """
    Divide data into input and output
    """
    #divide data into input and output
    self.x = self.data[self.xcols]
    self.y = self.data[self.ycols]
    
    #convert structured array to array
    x = np.array(x.tolist())
    y = np.array(y.tolist())
    return x, y

def rescale_log(self):
    """
    Scale the data to a logarithmic scale
    """
    if self.toXtype == 'flux':
        for col in self.xcols:
            if col.startswith(self.dust + self.inXtype):
                self.data[col] = np.log10(self.data[i])

    if self.
    self.y = rescale_log(y)

def fitscale(a, scaler=StandardScaler()):
    """
    Fit a scaler to data a. Returns fitted scaler
    """
    scaler.fit(a)
    return scaler

def rescale(a, scaler):
    """
    Transform the data a using standard or linear scaler
    """
    return scaler.transform(a)

def invscale(a, scaler):
    """
    Transform data a back to its original values
    """    
    return scaler.inverse_transform(a)

def remove_zero(x, y):
    """
    Remove galaxies with zero mass or zero flux
    Arguments:
        x       - galaxy fluxes (array)
        y       - galaxy stellar mass (array)
    Returns
        x       - galaxy fluxes (array)
        y       - galaxy stellar mass (array)   
    """
    y = y[~np.any(x == 0, axis=1)]
    x = x[~np.any(x == 0, axis=1)]
    x = x[~np.any(y == 0, axis=1)]
    y = y[~np.any(y == 0, axis=1)]
    return x, y

def remove_inf(x, y):
    """
    Remove galaxies with infinite values caused by log
    Arguments:
        x       - galaxy fluxes (array)
        y       - galaxy stellar mass (array)
    Returns
        x       - galaxy fluxes (array)
        y       - galaxy stellar mass (array)  
    """
    y = y[~np.isinf(x).any(axis=1)]
    x = x[~np.isinf(x).any(axis=1)]  
    x = x[~np.isinf(y).any(axis=1)]
    y = y[~np.isinf(y).any(axis=1)]
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
