import numpy as np
import matplotlib.pyplot as plt

#EAGLE settings
sim = 'RefL0050N0752' #simulation
cat = 'sdss' #catalogue to build model to
snapshots = np.arange(29) #redshift to use

dir = '/disks/strw9/vanweenen/mrp2/data/'
file = sim+'-'+cat+'.csv'

def read_data(loc):
    dtype=['<i8','<i8','<f8','<f8','<f8','<f8','<f8','<f8','<f8', '<f8']
    return np.genfromtxt(loc, delimiter=',', dtype=dtype, skip_header=15, names=True)

def select_redshift(data, s):
    """
    Selecting all galaxies from one redshift snapshot
    """    
    print("Selecting redshift..")    
    #select only data of a given redshift
    return data[np.where(data['z'] == data['z'][s])]

data = read_data(dir+file)

galaxies_per_z = []
for s in snapshots:
    data = select_redshift(data, s)
    gals = len(data)
    galaxies_per_z.append(gals)
    print("Snapshot ", s, " galaxies ", gals)

print(galaxies_per_z)
