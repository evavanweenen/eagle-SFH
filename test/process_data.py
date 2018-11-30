import numpy as np

sim = 'RefL0025N0376'
cat = 'sdss'
dir = '/disks/strw9/vanweenen/mrp2/test/'
fname = sim+'-'+cat+'.csv'

dtype=['<i8','<i8','<f8','<f8','<f8','<f8','<f8','<f8', '<f8']
data = np.genfromtxt(dir+fname, delimiter=',', dtype=dtype, skip_header=10, names=True)


def select_redshift(data, i):
    return data[np.where(data['z'] == data['z'][i])]

