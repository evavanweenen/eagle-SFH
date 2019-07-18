import numpy as np
from matplotlib import pyplot as plt

combs = combination_input(('u', 'g', 'r', 'i', 'z') + ('ug', 'ur', 'ui', 'uz', 'gr', 'gi', 'gz', 'ri', 'rz', 'iz'))

xcols_strings = [','.join(comb) for comb in combs]

loc = '/disks/strw9/vanweenen/mrp2/plots/'

scores = np.empty((len(combs), 3, 5))
cv = np.empty((len(combs), 5))
eagle_test = np.empty((len(combs), 5))
sdss = np.empty((len(combs), 5))

for i, comb in enumerate(combs):
    name = 'score_'+xcols_strings[i]+'_m_star_inp=nocolors_sampling=uniform.pdf.npy'
    
    s = np.load(loc+name)
    
    scores[i] = s
    cv[i] = s[0] #todo check
    eagle_test[i] = s[1]
    sdss = s[2]
    

score_names = ('MAE', '$R^2$', '$\bar{R}^2$', 'mean error', 'var error')
for i, name in enumerate(score_names):
    plt.figure()
    plt.bar(cv[i])
    plt.xticks(np.arange(len(xcols_strings)), fontsize=8)
    plt.gca().set_yticklabels(xcols_strings)
    plt.ylabel(name)    
