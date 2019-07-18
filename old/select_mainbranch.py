from eagle.io import *
from eagle.plot import *

sim = 'RefL0100N1504' 
cat = 'dusty-sdss-nomain'

dtype = ['<i8','<i8','<f8','<f8','<f8','<f8','<f8','<f8','<f8', '<f8']

#read data
data = read_data('', sim, cat, dtype=dtype, skip_header=15)

#reference redshift
z28 = np.min(data['z'])

#mainbranch array
dtype_mainbranch=[('topleafid', '<i8'), ('galid', '<i8'), ('z', '<f8'), ('m_star', '<f8'), ('sfr', '<f8'), ('dusty_sdss_u', '<f8'), ('dusty_sdss_g', '<f8'), ('dusty_sdss_r', '<f8'), ('dusty_sdss_i', '<f8'), ('dusty_sdss_z', '<f8')]
mainbranch = np.array([], dtype=dtype_mainbranch)

#select main branch
for ref in data[np.where(data['z'] == z28)]:
    for prog in data:
        if prog['galid'] <= ref['topleafid'] and prog['galid'] >= ref['galid']:
            mainbranch = np.append(mainbranch, prog)

#write data to file
write_data('', sim, 'dusty-sdss-main', mainbranch, fmt='%u,%u,%E,%E,%E,%E,%E,%E,%E,%E')
