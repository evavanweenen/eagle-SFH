import numpy as np 
import h5py 

dir = '/disks/strw9/vanweenen/mrp2/'
file = 'Ref100_histories.hdf5' 
hf = h5py.File(dir+file, 'r')  

galid = hf['Galaxies/GalaxyID'] 
histories = hf['Histories/SFR_vs_time'] 
redshift = hf['Histories/Z_vs_time']



