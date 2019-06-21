import numpy as np

def LOAD(inp, sampling):
    fluxes =  ('u', 'g', 'r', 'i', 'z') 
    if inp == 'nocolors': 
        colors = () 
    elif inp == 'subsetcolors': 
        colors = ('ug', 'gr', 'ri', 'iz') 
    elif inp == 'allcolors': 
        colors = ('ug', 'ur', 'ui', 'uz', 'gr', 'gi', 'gz', 'ri', 'rz', 'iz') 
    x = fluxes + colors 
    x_string = ','.join(x)

    path = '/disks/strw9/vanweenen/mrp2/plots/variance_output/'+str(sampling)+'_'+inp+'/score_'+x_string+'_m_star_inp='+inp+'_sampling='+str(sampling)+'.pdf.npy'
    return np.load(path).T


sampling=None
inp = 'nocolors'
print(LOAD(inp, sampling))


"""
def FNAME(inp, sampling, x_string): 
	return 'score_'+x_string+'_m_star_inp='+inp+'_sampling='+str(sampling)+'.pdf.npy' 
	
def XCOLS(inp): 
    fluxes =  ('u', 'g', 'r', 'i', 'z') 
    if inp == 'nocolors': 
        colors = () 
    elif inp == 'subsetcolors': 
        colors = ('ug', 'gr', 'ri', 'iz') 
    elif inp == 'allcolors': 
        colors = ('ug', 'ur', 'ui', 'uz', 'gr', 'gi', 'gz', 'ri', 'rz', 'iz') 
    x = fluxes + colors 
    x_string = ','.join(x)
    return x_string 

def LOAD(inp, sampling)
    loc = '/disks/strw9/vanweenen/mrp2/plots/variance_input/'+str(sampling)+'_'+inp+'/' 
    xcols_string = XCOLS(inp)
    fname = FNAME(inp, sampling, xcols_string) 
    np.load(loc+fname).T

LOAD(inp, sampling)
"""
