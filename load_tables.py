import numpy as np

folder = 'mass_brinchmann/'

np.set_printoptions(linewidth=200) 

def LOAD(inp, sampling):
    fluxes = ('u', 'g', 'r', 'i', 'z') 
    if inp == 'nocolors': 
        colors = () 
    elif inp == 'subsetcolors': 
        colors = ('ug', 'gr', 'ri', 'iz') 
    elif inp == 'allcolors': 
        colors = ('ug', 'ur', 'ui', 'uz', 'gr', 'gi', 'gz', 'ri', 'rz', 'iz') 
    x = fluxes + colors
    x_string = ','.join(x)

    path = '/disks/strw9/vanweenen/mrp2/plots/'+folder+str(sampling)+'_'+inp+'/score_'+x_string+'_m_star_inp='+inp+'_sampling='+str(sampling)+'.pdf.npy'
    return np.load(path).T

samplings=(None, 'random', 'uniform')
inp = ('nocolors', 'subsetcolors', 'allcolors')

tot_arr = np.empty((15,0))
for s in samplings:
    s_arr = np.empty((0,3))
    for i in inp:
        arr = LOAD(i, s)
        s_arr = np.concatenate((s_arr, arr))
    tot_arr = np.hstack((tot_arr, s_arr))
print(tot_arr)
