from .calc import *

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, append_fields

import h5py
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from astropy.cosmology import Planck13 as cosmo

class SDSS:
    def __init__(self, xcols, ycols, xtype, redshift):
        self.dir = '/disks/strw9/vanweenen/mrp2/data/'
        self.file = 'SDSS_DR9.hdf5'
        self.xtype = xtype
        self.xcols = xcols 
        self.ycols = ycols
        self.datacols = [xtype + '_' + f for f in ('u', 'g', 'r', 'i', 'z')] + ycols + ['Redshift']
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

    def preprocess(self, colors, frac = 5e-3):
        self.read_data()

        #select only galaxies of given redshift
        self.select_redshift(frac)
        
        #to magnitude
        if self.xtype == 'flux':
            to_magnitude(self, self.xtype, luminosity_distance_sdss)

        #add colors
        add_colors(self, self.xtype+'_', colors)

        #select only columns necessary
        self.datacols = self.xcols + self.ycols
        select_cols(self)

        #divide data into x and y
        x, y = divide_input_output(self)

        #remove galaxies with zero y
        x, y = remove_zero(x, y)
        
        #remove galaxies with -1 y
        x, y = remove_minone(x, y)
        
        #remove galaxies with nan y
        x, y = remove_nan(x, y)

        #remove infinite values caused by logarithmic scaling of x
        self.x, self.y = remove_inf(x, y)

    def scaling(self, eagle):
        self.y = self.y.reshape(-1,1)
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1,1)
        self.x = eagle.xscaler.transform(self.x)
        self.y = eagle.yscaler.transform(self.y)

    def postprocess(self, eagle, x, y, y_pred):
        x = eagle.xscaler.inverse_transform(x)
        y = eagle.yscaler.inverse_transform(y)
        y_pred = eagle.yscaler.inverse_transform(y_pred)
        return x, y, y_pred

class EAGLE:
    def __init__(self, fol, sim, cat, dust, snap, redshift, seed, xcols, ycols, xtype='flux'):
        self.dir = '/disks/strw9/vanweenen/mrp2/data/'            
        self.fol = fol                      # location of file
        self.sim = sim                      # EAGLE simulation (e.g. 'RefL0100N1504')
        self.cat = cat                      # fluxes catalogue (e.g. 'sdss')
        self.dust = dust                    # 'dusty' or ''
        self.snap = snap                    # snapshot
        self.redshift = redshift            # redshift of snapshot
        self.path = self.dir + self.fol + self.sim + '-' + self.dust + self.cat + '-snap' + str(self.snap) + '.csv'
        self.seed = seed                    # random seed
        self.xtype = xtype                  # (string) input type that you read from file ('flux' or (absolute!) 'magnitude')
        self.xcols = xcols                  # (list) column names of input
        self.ycols = ycols                  # (list) column names of output
        self.datacols = self.xcols + self.ycols + ['subgroup']   # data columns
        
        np.random.seed(self.seed)

    """
    def read_histories(self, z_range, interval=1000):
        f = h5py.File(self.dir + 'Ref100_histories.hdf5', 'r')
        galid = f.get('Galaxies').get('GalaxyID')[()]
        SFR = f.get('Histories').get('SFR_vs_time')[()]
        age = f.get('Histories').get('AgeBinEdges')[()]
        for z in z_range:
            edge = np.where(age == np.around(cosmo.age(z).value, decimals=3))[0][0]
            half_int = int(interval/2)
            if edge > len(age) - half_int:
                edge = len(age) - half_int
            elif edge < half_int:
                edge = half_int
            np.average(SFR[:,edge-half_int:edge+half_int], axis=1)
    """         

    def read_data(self, dtype=2*['<i8']+['<f8']+['<i8']+12*['<f8'], skip_header=21):
        """
        Read csv file with data from eagle
        """
        print("Reading data with of simulation %s, catalogue %s"%(self.sim, self.dust+self.cat))
        self.data = np.genfromtxt(self.path, delimiter=',', names=True, dtype=dtype, skip_header=skip_header)
    
    def preprocess(self, colors):
        """
        colors          - (tuple) colors to add to data (e.g. ('ug', 'gr', 'ri', 'iz'))
        """
        #read data
        self.read_data()

        #convert fluxes to magnitudes if input is fluxes
        if self.xtype == 'flux':
            to_magnitude(self, self.dust+self.xtype, luminosity_distance_eagle)

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

    def scaling(self):
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1,1)
        self.y = self.y.reshape(-1,1)
        self.xscaler = MinMaxScaler(feature_range=(-1,1)).fit(self.x)#StandardScaler())#
        self.yscaler = MinMaxScaler(feature_range=(-1,1)).fit(self.y)#StandardScaler())#
        self.x = self.xscaler.transform(self.x)
        self.y = self.yscaler.transform(self.y)

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

def sample(eagle, sdss, sampling=None, bins=10, count=100, N=500):
    """
    eagle           - (class) eagle class with loaded eagle data
    sdss            - (class) sdss class with loaded sdss data
    sampling        - (string) the type of sampling to perform (None / 'random' / 'uniform')
    bins            - (int) the number of bins for the uniform mass sampling
    count           - (int) the number of galaxies per bin
    N               - (int) the size of the random sample
    """
    if sampling == 'uniform':
        #uniform_mass_sampling(eagle, ref_pre=sdss, bins=bins, count=count, cs_arr=True)
        #uniform_mass_sampling(sdss, ref_post=eagle, bins=bins, count=int(count*0.2), cs_arr=False)
        both_uniform_mass_sampling(eagle, sdss, bins=bins, count=count)
    if sampling == 'random' or sampling == 'uniform':
        random_sampling(eagle, N)
        random_sampling(sdss, int(N*0.2), cs_arr=False)

def to_magnitude(self, pre, DL):
    """
    If input are fluxes, convert fluxes to (absolute!) magnitudes, using ealge.calc functions.
    """
    for col in self.data.dtype.names:
        if col.startswith(pre):
            self.data[col] = app_to_abs_mag(flux_to_magAB(self.data[col]), DL(self.redshift))

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

def random_sampling(self, N, cs_arr=True):
    """
    Take a random sample of size N from the data
    """
    perm = np.random.permutation(self.x.shape[0])[:N] #take first N numbers of a random permutation
    self.x = self.x[perm]
    self.y = self.y[perm]
    if cs_arr:
        self.cs = self.cs[perm]

def consecutive_trues(remain):
    split_remain_bool = np.array([key for key, group in itertools.groupby(remain)])
    split_remain = np.array([sum(1 for _ in group) for key, group in itertools.groupby(remain)])
    split_remain_false0 = np.array([sum(1 for _ in group) if key else 0 for key, group in itertools.groupby(remain)])
    max_place = np.argmax(split_remain_false0)
    new_remain = []
    for i in range(len(split_remain)):
        if i == max_place:
            new_remain += split_remain[i]*[split_remain_bool[i]]
        else:
            new_remain += split_remain[i]*[False]
    return new_remain

def both_uniform_mass_sampling(eagle, sdss, bins=10, count=125, N=625, perc_train=0.8, cs_arr=True, ensure_consec=True):
    """
    Uniform mass sampling for both eagle and sdss
    """
    count_sdss = round(count*(1-perc_train))
    
    eagle.y = eagle.y.reshape(-1,1)
    sdss.y = sdss.y.reshape(-1,1)
    if eagle.x.ndim == 1:
        eagle.x = eagle.x.reshape(-1,1)
    if sdss.x.ndim == 1:
        sdss.x = sdss.x.reshape(-1,1)
    eagle.hist, eagle.edges = np.histogram(eagle.y, bins=bins)
    sdss.hist, sdss.edges = np.histogram(sdss.y, bins=eagle.edges)

    #array with lower edges and upper edges of bins that are above the minimal count
    remain = (eagle.hist > count) & (sdss.hist > count_sdss)
    
    if ensure_consec:
        #make sure that only consecutive trues remain
        remain = consecutive_trues(remain)
    
    #make sure not much random sampling has to be done after the uniform sampling    
    n_remain = len(np.where(remain)[0])
    count_remain = int(np.ceil(N/n_remain)) #ideally number of galaxies per bin
    if int(np.ceil(count_remain/5)*5) < count:
        count = int(np.ceil(count_remain/5)*5)
        count_sdss = round(count*(1-perc_train))
       
    lower_edges = eagle.edges[np.insert(remain, len(eagle.hist), False)]
    upper_edges = eagle.edges[np.insert(remain, 0, False)]
    hist_eagle = eagle.hist[remain]
    hist_sdss = sdss.hist[remain]
    
    #create new x and y data consisting of an equal number of galaxies per mass bin
    #for each bin, randomly shuffle the data in the bin, and then keep the first count galaxies
    x_equal_eagle = np.empty((count*len(hist_eagle), eagle.x.shape[1]))
    y_equal_eagle = np.empty((count*len(hist_eagle), eagle.y.shape[1]))
    x_equal_sdss = np.empty((count_sdss*len(hist_sdss), sdss.x.shape[1]))
    y_equal_sdss = np.empty((count_sdss*len(hist_sdss), sdss.y.shape[1]))
    if cs_arr:
        cs_equal = np.zeros((count*len(hist_eagle)), dtype=bool)
    for i in range(len(lower_edges)):
        perm_eagle = np.random.permutation(hist_eagle[i])[:count]
        perm_sdss = np.random.permutation(hist_sdss[i])[:count_sdss]
        mask_eagle = (eagle.y >= lower_edges[i]) & (eagle.y < upper_edges[i])
        mask_sdss = (sdss.y >= lower_edges[i]) & (sdss.y < upper_edges[i])
        mask_eagle = mask_eagle.reshape(len(mask_eagle),)
        mask_sdss = mask_sdss.reshape(len(mask_sdss),)
        x_equal_eagle[i*count:(i+1)*count] = eagle.x[mask_eagle][perm_eagle]
        y_equal_eagle[i*count:(i+1)*count] = eagle.y[mask_eagle][perm_eagle]
        x_equal_sdss[i*count_sdss:(i+1)*count_sdss] = sdss.x[mask_sdss][perm_sdss]
        y_equal_sdss[i*count_sdss:(i+1)*count_sdss] = sdss.y[mask_sdss][perm_sdss]
        if cs_arr:
            cs_equal[i*count:(i+1)*count] = eagle.cs[mask_eagle][perm_eagle]
    eagle.x = x_equal_eagle ; sdss.x = x_equal_sdss
    eagle.y = y_equal_eagle ; sdss.y = y_equal_sdss
    if cs_arr:
        eagle.cs = cs_equal

def uniform_mass_sampling(self, bins=10, count=100, ref_pre=None, ref_post=None, perc_train=0.8, cs_arr=True):
    """
    Uniform mass sampling
    Supply ref_pre and ref_post if you want two samples to have the same mass distribution
    """
    if ref_post != None:
        bins = ref_post.edges
    self.y = self.y.reshape(-1,1)
    if self.x.ndim == 1:
        self.x = self.x.reshape(-1,1)
    self.hist, self.edges = np.histogram(self.y, bins=bins)
    if ref_pre != None:
        ref_pre.y = ref_pre.y.reshape(-1,1)
        ref_pre.hist, ref_pre.edges = np.histogram(ref_pre.y, bins=self.edges)

    #array with lower edges and upper edges of bins that are above the minimal count
    remain = (self.hist > count)
    if ref_pre != None:
        remain &= (ref_pre.hist > count*(1-perc_train))
    if ref_post != None:
        remain &= (ref_post.hist > count/(1-perc_train))
    lower_edges = self.edges[np.insert(remain, len(self.hist), False)]
    upper_edges = self.edges[np.insert(remain, 0, False)]
    hist = self.hist[remain]

    #create new x and y data consisting of an equal number of galaxies per mass bin
    #for each bin, randomly shuffle the data in the bin, and then keep the first count galaxies
    x_equal = np.empty((count*len(hist), self.x.shape[1]))
    y_equal = np.empty((count*len(hist), self.y.shape[1]))
    if cs_arr:
        cs_equal = np.zeros((count*len(hist)), dtype=bool)
    for i in range(len(lower_edges)):
        perm = np.random.permutation(hist[i])[:count]
        mask = (self.y >= lower_edges[i]) & (self.y < upper_edges[i])
        mask = mask.reshape(len(mask),)
        x_equal[i*count:(i+1)*count] = self.x[mask][perm]
        y_equal[i*count:(i+1)*count] = self.y[mask][perm]
        if cs_arr:
            cs_equal[i*count:(i+1)*count] = self.cs[mask][perm]
    self.x = x_equal ; self.y = y_equal
    if cs_arr:
        self.cs = cs_equal

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

def remove_minone(x, y):
    """
    Remove galaxies with zero mass    
    """
    x = x[y != -1.]
    y = y[y != -1.]
    return x, y

def remove_inf(x, y):
    """
    Remove galaxies with infinite values caused by 0's in logarithmic scaling
    """
    if x.ndim != 1:
        y = y[~np.isinf(x).any(axis=1)]
        x = x[~np.isinf(x).any(axis=1)]
    else:
        y = y[~np.isinf(x)]
        x = x[~np.isinf(x)]
    return x, y

def remove_nan(x, y):
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
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
