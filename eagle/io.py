import numpy as np
from sklearn.preprocessing import MinMaxScaler

dir = '/disks/strw9/vanweenen/mrp2/data/'

def read_data(folder, sim, cat, **kwargs):
    """
    Read csv file with data from eagle
    Arguments:
        dir         - location of file
        sim         - simulation
        cat         - fluxes catalogue
    Returns:
        data (numpy ndarray)
    """
    print("Reading data with of simulation %s, catalogue %s"%(sim, cat))    
    file = sim+'-'+cat+'.csv'    
    return np.genfromtxt(dir+folder+file, delimiter=',', names=True, **kwargs)

def select_redshift(data, snapshot):
    """
    Select only data of a given redshift
    Arguments:
        data        - data of galaxies (numpy ndarray)
        snapshot    - snapshot to use (28 is at z=0, 0 is highest z)
    Returns:
        data        - only data of a given redshift (numpy ndarray)
    """
    print("Selecting data of a given redshift at snapshot %s .."%snapshot)

    #TODO: DON'T USE THIS FUNCTION, IT IS NOT CORRECT  
      
    return data[np.where(data['z'] == data['z'][snapshot])]

def divide_input_output(data, xcols, ycols):
    """
    Divide data into input and output
    Arguments:
        data        - (numpy ndarray)
        xcols       - column names of input (list)
        ycols       - column names of output (list)
    Returns:
        x, y        - input, output (list, list)    
    """
    #divide data into input and output
    x = data[xcols]
    y = data[ycols]
    
    #convert structured array to array
    x = np.array(x.tolist())
    y = np.array(y.tolist())
    return x, y

def rescale_lin(x, y):
    """
    Scale the data to a linear scale
    """
    #rescale the data to values between 0 and 1
    xscaler = MinMaxScaler(feature_range=(0,1))
    x = xscaler.fit_transform(x)
    yscaler = MinMaxScaler(feature_range=(0,1))
    y = yscaler.fit_transform(y)
    return x, y, xscaler, yscaler

def rescale_log(x, y):
    """
    Scale the data to a logarithmic scale
    """
    #take logarithm of data
    x = np.log10(x)
    y = np.log10(y)
    return x, y

def invscale_lin(data, scaler):
    """
    Transform x and y back to their original values
    """    
    data = scaler.inverse_transform(data)
    return data

def perm_train_test(x, y, length, perc_train=.8):
    """
    Apply a random permutation to the data and put perc_train percent in the training set and the remainder in the test set
    """    
    print("Permuting data..")    
    #permute the data into the training and test set
    
    perm = np.random.choice([True, False], length, p=[perc_train, 1-perc_train])
    return x[perm,:], y[perm], x[np.invert(perm),:], y[np.invert(perm)]
    
def data_ndarray(x, y, xcols, ycols):
    """
    Convert x and y back to a numpy ndarray
    """
    #convert array to structured array
    x = np.core.records.fromarrays(x.transpose(), names=xcols, formats=5*['<f8'])
    y = np.core.records.fromarrays(y.transpose(), names=ycols, formats=['<f8'])
    
    #merge x and y back together
    data = np.lib.recfunctions.merge_arrays([x,y], flatten=True)
    return data
