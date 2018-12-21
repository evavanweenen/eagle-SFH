from eagle.io import *
from eagle.plot import *

from scipy.sparse import coo_matrix, csc_matrix

fol = 'dustysdss_mstar_mainbranch/'
sims = ['RefL0025N0376','RefL0025N0752', 'RecalL0025N0752', 'RefL0050N0752','RefL0100N1504']
cat = 'dusty-sdss'
L = [25, 25, 25, 50, 100] #Mpc

dtype = ['<i8','<i8','<f8','<f8','<f8','<f8','<f8','<f8','<f8', '<f8']

sfrs = []
zs = []
for sim in sims:
    file = sim+'-'+cat+'.csv'

    dtype=['<i8','<i8','<f8','<f8','<f8']
    data = read_data(fol, sim, cat, dtype=dtype, skip_header=15)
    
    # topleafid_index is index of all unique topleafids
    # topleafids[topleafid_inverse] = data['topleafid']
    topleafids, topleafid_index, topleafid_inverse = np.unique(data['topleafid'], return_index=True, return_inverse=True)
    # z is sorted array of all unique z
    # z[z_inverse] = data['z']
    z, z_inverse = np.unique(data['z'], return_inverse=True)

    # Create sparse matrix with sfr for each topleafid-z combination where topleafid_inverse give the locations of the respective
    # topleafid and z_inverse gives the location of the respective redshift. The rows of this matrix are actually the star
    # formation history of one galaxy. The sum of a column of this matrix is the total SFR of one redshift.
    sfr = csc_matrix(coo_matrix((data['sfr'], (topleafid_inverse, z_inverse))))

    print("nr of redshifts: ", len(z), " nr of galaxies in top level: ", len(topleafid_index))
    
    sfrs.append(sfr)
    zs.append(z)



plot_sfr_z(sims, L, sfrs, zs)

"""
plt.figure()
plt.title('Star formation history of one galaxy')
for i, sim in enumerate(sims):
    galaxy_index = 0
    sfr_1gal = sfrs[i].getrow(galaxy_index).toarray()[0]
    plt.plot(zs[i], sfr_1gal/(L[i]**3), label=sim+'; topleafid=%s'%data['topleafid'][topleafid_index[galaxy_index]])
plt.xscale('log')
plt.xlim(6*10**-2, 10**1)
plt.xlabel('z')
plt.ylabel('$SFR$ ($M_{\odot} yr^{-1} Mpc^{-3}$)')
plt.legend()
plt.savefig('sfh_1gal.pdf')
plt.show()
"""












