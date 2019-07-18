from eagle.io import *
from eagle.plot import *
from eagle.nn import *

import numpy as np
from copy import copy
import tensorflow as tf
import shap

seed = 7
np.random.seed(seed) #fix seed for reproducibility
tf.set_random_seed(seed)

plotting = True

inp = 'nocolors'#'subsetcolors'#'allcolors'#

input_variance=True #TODO
input_uniform=False #TODO

#EAGLE settings
fol = ''
sim = 'RefL0100N1504'   # simulation
cat = 'sdss'            # catalogue to build model to
dust = 'dusty'          # '' if no dust
snap = 27               # snapshot to use
redshift = 1.0063854e-01#0.1# redshift of snapshot 
#redshift_array = np.flip(np.array([2.2204460e-16, 1.0063854e-01, 1.8270987e-01, 2.7090108e-01, 3.6566857e-01, 5.0310730e-01, 6.1518980e-01, 7.3562960e-01, 8.6505055e-01, 1.0041217e+00, 1.2593315e+00, 1.4867073e+00, 1.7369658e+00, 2.0124102e+00, 2.2370370e+00, 2.4784133e+00, 3.0165045e+00, 3.5279765e+00, 3.9836636e+00, 4.4852138e+00, 5.0372367e+00, 5.4874153e+00, 5.9711623e+00, 7.0495663e+00, 8.0746160e+00, 8.9878750e+00, 9.9930330e+00]))
#redshift = redshift_array[snap - 2]

fluxes =  ('u', 'g', 'r', 'i', 'z')
if inp == 'nocolors':
    colors = ()
elif inp == 'subsetcolors':
    colors = ('ug', 'gr', 'ri', 'iz')
elif inp == 'allcolors':
    colors = ('ug', 'ur', 'ui', 'uz', 'gr', 'gi', 'gz', 'ri', 'rz', 'iz')

xcols = fluxes + colors
xcols_string = ','.join(xcols)

#eagle
eagle_xtype = 'flux'
eagle_xcols = [dust + eagle_xtype + '_' + cat + '_' + f for f in xcols]
eagle_ycols = ['m_star']#['sfr']#

#sdss
sdss_xtype = 'flux'
sdss_xcols = [sdss_xtype + '_' + f for f in xcols]
sdss_ycols = ['Mstellar_median']#['SFR_median']#

#input output plot
#xnames=[dust + ' ' + cat + ' ' + i for i in list(fluxes) + [c[0] + '-' + c[1] for c in colors]]
xnames= list(fluxes) + [c[0] + '-' + c[1] for c in colors]
ynames=['$\log_{10} M_{*} (M_{\odot})$']#['$\log_{10}$ SFR $(M_{\odot} yr^{-1})$']#

#preprocessing
cs = False ; cs_mask = None #central satellite analysis (only change cs)
sampling = 'uniform' ; bins = 10 ; count = 50 ; N = 625 ; scaling = True if sampling == None else False #both same uniform mass distribution #TODO

"""
#ML settings GridSearch
h_nodes = [20, 50, 50]
activation = ['tanh', 'linear', 'tanh', 'linear']
optimizer = 'Adamax'
"""
if inp == 'nocolors':
    #Hyperas no colors
    h_nodes = [40, 50, 30]
    activation = ['linear', 'tanh', 'tanh', 'linear']
elif inp == 'subsetcolors':
    #Hyperas subset colors
    h_nodes = [50, 50, 20]
    activation = ['relu', 'tanh', 'linear', 'linear']
elif inp == 'allcolors':
    #Hyperas all colors
    h_nodes = [30, 30, 50]
    activation = ['linear', 'tanh', 'relu', 'linear']

optimizer = 'Adam'
dropout = [0., 0., 0.]
perc_train = .8 #todo: this is not used anywhere, give it to io
loss = 'mean_absolute_error'
epochs = 15
batch_size = 128

def cross_validation(X, Y, K, seed):
    kf = KFold(n_splits=K, random_state=seed)
    X_train = [] ; Y_train = []; X_val = []; Y_val = []
    for idx_train, idx_val in kf.split(X):
        X_train.append(X[idx_train])
        Y_train.append(Y[idx_train])
        X_val.append(X[idx_val])
        Y_val.append(Y[idx_val])       
    return X_train, Y_train, X_val, Y_val

def add_scores(score, x, y, y_pred):
    y.reshape(len(y),) ; y_pred.reshape(len(y_pred),)
    score[1] = r2_score(y, y_pred)
    return np.append(score, [adjusted_R_squared(score[1], x.shape[1], len(y)), np.mean(y_pred - y), np.var(y_pred - y)])

#------------------------------------------ Read data --------------------------------------------------------------
#read eagle data
eagle = EAGLE(fol, sim, cat, dust, snap, redshift, seed)
eagle.xtype = eagle_xtype
eagle.xcols = eagle_xcols
eagle.ycols = eagle_ycols
eagle.datacols = eagle.xcols + eagle.ycols + ['subgroup']   # data columns
eagle.perc_train = perc_train
eagle.read_data()

#read sdss data and select only galaxies of redshift
sdss = SDSS(sdss_xcols, sdss_ycols, sdss_xtype, redshift)
sdss.datacols += ['e_' + sdss.xtype + '_' + f for f in ('u', 'g', 'r', 'i', 'z')]
sdss.read_data()
sdss.select_redshift(5e-3)

for col in sdss.xcols[:5]:
    sdss.data = sdss.data[sdss.data[col] != 0.]

#calculate average error per flux bin
flux_bins = 20; flux_count=25
errors = np.empty((len(sdss.xcols[:5]), flux_bins))
edges = np.empty((len(sdss.xcols[:5]), flux_bins+1))
for i, col in enumerate(sdss.xcols[:5]):
    hist_sdss, edges[i] = np.histogram(np.log10(sdss.data[col]), bins=flux_bins)
    hist_eagle = np.histogram(np.log10(eagle.data[eagle.xcols[i]]), bins=edges[i])[0]
    remain = (hist_sdss > flux_count) & (hist_eagle > flux_count)
    print("Number of remaining bins for " + col + ":", len(np.where(remain)[0]))

    for j in range(flux_bins):
        mask_eagle = (eagle.data[eagle.xcols[i]] >= 10.**edges[i,j]) & (eagle.data[eagle.xcols[i]] < 10.**edges[i,j+1])
        mask_sdss = (np.log10(sdss.data[col]) >= edges[i,j]) & (np.log10(sdss.data[col]) < edges[i,j+1])
        if remain[j]:
            if input_variance:
                error = np.average(sdss.data['e_'+col][mask_sdss])
                eagle.data[eagle.xcols[i]][mask_eagle] = np.random.normal(eagle.data[eagle.xcols[i]][mask_eagle], error/2)
        else:
            eagle.data[eagle.xcols[i]][mask_eagle] = np.nan
            sdss.data[col][mask_sdss] = np.nan

#remove nans
for col in eagle.xcols[:5]:
    eagle.data = eagle.data[~np.isnan(eagle.data[col])]

for col in sdss.xcols[:5]:
    sdss.data = sdss.data[~np.isnan(sdss.data[col])]

#preprocessing of eagle
to_magnitude(eagle, eagle.dust+eagle.xtype, luminosity_distance_eagle)
if len(colors) != 0:
    add_colors(eagle, eagle.dust+eagle.xtype+'_'+eagle.cat+'_', colors)
select_cols(eagle)
rescale_log(eagle)
eagle.x, eagle.y = divide_input_output(eagle)
eagle.cs = (eagle.data['subgroup'] == 0)

#preprocessing of sdss
to_magnitude(sdss, sdss.xtype, luminosity_distance_sdss)
add_colors(sdss, sdss.xtype+'_', colors)
sdss.datacols = sdss.xcols + sdss.ycols
select_cols(sdss)
sdss.x, sdss.y = divide_input_output(sdss)
sdss.x, sdss.y = remove_zero(sdss.x, sdss.y)
sdss.x, sdss.y = remove_inf(sdss.x, sdss.y)

#uniform selection after preprocessing
if input_uniform:
    edges_z = np.flip(app_to_abs_mag(flux_to_magAB(10.**edges[4]), luminosity_distance_sdss(redshift)))

    hist_sdss = np.histogram(sdss.x[:,4], bins=edges_z)[0]
    hist_eagle = np.histogram(eagle.x[:,4], bins=edges_z)[0]
    remain_z = (hist_sdss > flux_count) & (hist_eagle > flux_count)
    
    sdss_equal_x = copy(sdss.x) ; eagle_equal_x = copy(eagle.x)
    sdss_equal_y = copy(sdss.y) ; eagle_equal_y = copy(eagle.y)
    
    sdss_equal_x[:] = np.nan ; eagle_equal_x[:] = np.nan
    sdss_equal_y[:] = np.nan ; eagle_equal_y[:] = np.nan
    
    for j in range(flux_bins):
        if remain_z[j]:
            mask_eagle = (eagle.x[:,4] >= edges_z[j]) & (eagle.x[:,4] < edges_z[j+1])
            mask_sdss = (sdss.x[:,4] >= edges_z[j]) & (sdss.x[:,4] < edges_z[j+1])
            perm_sdss = np.random.permutation(hist_sdss[j])[:flux_count]
            perm_eagle = np.random.permutation(hist_eagle[j])[:flux_count]
            sdss_equal_x[j*flux_count:(j+1)*flux_count] = sdss.x[mask_sdss][perm_sdss]
            sdss_equal_y[j*flux_count:(j+1)*flux_count] = sdss.y[mask_sdss][perm_sdss]
            eagle_equal_x[j*flux_count:(j+1)*flux_count] = eagle.x[mask_eagle][perm_eagle]
            eagle_equal_y[j*flux_count:(j+1)*flux_count] = eagle.y[mask_eagle][perm_eagle]
        
    sdss.x = sdss_equal_x ; eagle.x = eagle_equal_x
    sdss.y = sdss_equal_y ; eagle.y = eagle_equal_y
    
    #remove nans
    for i in range(len(eagle.xcols)):
        eagle.y = eagle.y[~np.isnan(eagle.x[:,i])]
        sdss.y = sdss.y[~np.isnan(sdss.x[:,i])]
        eagle.x = eagle.x[~np.isnan(eagle.x[:,i])]
        sdss.x = sdss.x[~np.isnan(sdss.x[:,i])]

if scaling:
    eagle.scaling()
    sdss.scaling(eagle)
x_train, x_test, y_train, y_test = train_test_split(eagle.x, eagle.y, test_size=1-perc_train, random_state=eagle.seed, shuffle=True)    

print("EAGLE: len(eagle.x) = %s ; len(x_train) = %s ; len(x_test) = %s ; len(sdss.x) = %s"%(len(eagle.x), len(x_train), len(x_test), len(sdss.x)))

if sampling == 'uniform':
    uniform_mass_sampling(eagle, ref_pre=sdss, count=100, cs_arr=True)
    uniform_mass_sampling(sdss, ref_post=eagle, count=int(100*0.2), cs_arr=False) #TODO

if sampling == 'uniform' or sampling == 'random':
    random_sampling(eagle, N)
    random_sampling(sdss, int(N*0.2), cs_arr=False) #TODO
    eagle.scaling()
    sdss.scaling(eagle)
    x_train, x_test, y_train, y_test, eagle.cs_train, eagle.cs_test = train_test_split(eagle.x, eagle.y, eagle.cs, test_size=1-perc_train, random_state=seed, shuffle=True)
    print("EAGLE: len(eagle.x) = %s ; len(x_train) = %s ; len(x_test) = %s ; len(sdss.x) = %s"%(len(eagle.x), len(x_train), len(x_test), len(sdss.x)))


if plotting:
    edges_x =  np.flip(app_to_abs_mag(flux_to_magAB(10.**edges), luminosity_distance_sdss(redshift)), axis=1)
    edges_x = eagle.xscaler.transform(np.vstack((edges_x, np.zeros((len(sdss.xcols)-5, edges_x.shape[1])))).T).T
    for i in np.arange(5, len(sdss.xcols)):
        min_x = np.amin(sdss.x[:,i]) ; max_x = np.amax(sdss.x[:,i])
        if min_x < -2.: min_x = -2.
        if max_x > 2.: max_x = 2.
        edges_x[i] = np.linspace(min_x, max_x, edges_x.shape[1])
    if sampling == 'uniform':
        edges_y = eagle.yscaler.transform(eagle.edges.reshape(-1,1)).T[0]
    else:
        edges_y = 7
    plot_data = PLOT_DATA(xnames+ynames, sim=sim, snap=snap, N=len(sdss.y), inp=inp, sampling=str(sampling))
    plot_data.hist_data(('eagle-train', 'eagle-test', 'sdss-total'), [np.hstack((x_train, y_train)), np.hstack((x_test, y_test)), np.hstack((sdss.x, sdss.y))], edges_y, edges_x, xlim=[-1.3,1.3], ylim=[-1.5,1.1])
"""
    plot_data.datanames = xnames+[ynames[0].split('(')[0]+'$']
    plot_data.statistical_matrix(x_test, y_test, ['eagle'], simple=True)
    plot_data.statistical_matrix(sdss.x, sdss.y, ['sdss'], simple=True)


input_size = len(x_train[0])
output_size = len(y_train[0])
nodes = [input_size] + h_nodes + [output_size]
print("nodes in the network: ", nodes)

#read hyperparameters
nn = NN(input_size, output_size, h_nodes, activation, dropout, loss, epochs, batch_size, optimizer)
new_result = AdditionalValidationSets([(sdss.x, sdss.y, 'val2')])
callbacks = [new_result]

#------------------------------------------Cross-validate on EAGLE--------------------------------------------------------------
K = 5

#crossvalidation data
X_train, Y_train, X_val, Y_val = cross_validation(x_train, y_train, K, seed)

cv_score = np.empty((K, 5))
for i in range(K):
    #make architecture of the network
    nn.MLP_model()
    result = nn.model.fit(X_train[i], Y_train[i], batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_val[i], Y_val[i]), callbacks=callbacks) 
    cv_score[i,:2] = nn.model.evaluate(X_val[i], Y_val[i], verbose=0)
    y_pred = nn.model.predict(X_val[i]) #predict
    cv_score[i] = add_scores(cv_score[i,:2], X_val[i], Y_val[i], y_pred) #add scores
avg_cv_score = np.average(cv_score, axis=0)

print("Cross-validated errors (EAGLE): MAE = %.3e ; R^2 = %.3f ; adjusted R^2 = %.3f ; error_mean = %.3e ; error_sigma = %.3e"%tuple(avg_cv_score))

#------------------------------------------Train and test on EAGLE--------------------------------------------------------------
#train, evaluate, predict, postprocess
nn.MLP_model()
result = nn.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test), callbacks=callbacks)
score = nn.model.evaluate(x_test, y_test, verbose=0) #evaluate
y_pred = nn.model.predict(x_test) #predict
score = add_scores(score, x_test, y_test, y_pred) #add scores
print("Errors (EAGLE): MAE = %.3e ; R^2 = %.3f ; adjusted R^2 = %.3f ; error_mean = %.3e ; error_sigma = %.3e"%tuple(score))

#central satellites analysis
if cs == True:
    cs_mask = eagle.cs_test
    score_c = nn.model.evaluate(x_test[cs_mask], y_test[cs_mask], verbose=0)
    score_s = nn.model.evaluate(x_test[~cs_mask], y_test[~cs_mask], verbose=0)
    score_c = add_scores(score_c, x_test[cs_mask], y_test[cs_mask], y_pred[cs_mask])
    score_s = add_scores(score_s, x_test[~cs_mask], y_test[~cs_mask], y_pred[~cs_mask])
    print("Errors (EAGLE) centrals: MAE = %.3e ; R^2 = %.3g ; adjusted R^2 = %.3g ; error_mean = %.3e ; error_sigma = %.3e"%tuple(score_c))
    print("Errors (EAGLE) satellites: MAE = %.3e ; R^2 = %.3g ; adjusted R^2 = %.3g ; error_mean = %.3e ; error_sigma = %.3e"%tuple(score_s))

#------------------------------------------Compare with SDSS--------------------------------------------------------------
#evaluate, predict, postprocess
sdss_score = nn.model.evaluate(sdss.x, sdss.y, verbose=0) #evaluate
sdss.ypred = nn.model.predict(sdss.x) #predict
sdss_score = add_scores(sdss_score, sdss.x, sdss.y, sdss.ypred)
print("Errors (SDSS): MAE = %.3e ; R^2 = %.3g ; adjusted R^2 = %.3g ; error_mean = %.3e ; error_sigma = %.3e"%tuple(sdss_score))

np.save('/disks/strw9/vanweenen/mrp2/plots/score_'+xcols_string+'_'+eagle_ycols[0]+'_inp='+inp+'_sampling='+str(sampling)+'.pdf', np.vstack((avg_cv_score, score, sdss_score)))

#------------------------------------------Calculate shapley values--------------------------------------------------------------
explainer = shap.DeepExplainer(nn.model, x_train)
shap_eagle = explainer.shap_values(x_test)[0]
shap_sdss = explainer.shap_values(sdss.x)[0]

shapdir = '/disks/strw9/vanweenen/mrp2/plots/mlp_dustyflux_mstar/feature_importance/shap/tests_linearfunc/one_coeff/'

plot_shap = PLOT_SHAP(xnames, xcols_string, inp, str(sampling))
plot_shap.summary_plot(x_test, shap_eagle, 'eagle', shap_sdss, 'sdss', plottype = 'bar')
plot_shap.summary_plot(x_test, shap_eagle, 'eagle', plottype = 'dot', dotsize=10)

#------------------------------------------Plotting--------------------------------------------------------------
x_test, y_test, y_pred = eagle.postprocess(x_test, y_test, y_pred) #postprocessing

#plot
if plotting:
    plot = PLOT_NN(eagle, nn, xnames, ynames, dataname='eagle', N=len(y_test), xcols_string=xcols_string, inp=inp, sampling=str(sampling), score=score)#[mse[19], r2[19]]
    plot.plot_learning_curve(new_result)
    plot.plot_input_output(x_test, y_test, y_pred, 'scatter') #scatter, contour
    plot.plot_true_predict(y_test, y_pred, 'scatter', cs_mask) #scatter, hexbin, contour, contourf
    plot.plot_output_error(y_test, y_pred, 'contourf', cs_mask) #scatter, hexbin, contour

sdss.x, sdss.y, sdss.ypred = sdss.postprocess(eagle, sdss.x, sdss.y, sdss.ypred) #postprocessing

#plot
if plotting:
    plot = PLOT_NN(eagle, nn, xnames, ynames, dataname='sdss', N=len(sdss.y), xcols_string=xcols_string, inp=inp, sampling=str(sampling), score=sdss_score)#[mse[19], r2[19]]
    plot.plot_input_output(sdss.x, sdss.y, sdss.ypred) #scatter, contour
    plot.plot_true_predict(sdss.y, sdss.ypred, 'scatter') #scatter, hexbin, contour, contourf
    plot.plot_output_error(sdss.y, sdss.ypred, 'contourf', ylim=(-0.15, 0.3)) #scatter, hexbin, contour
    
"""    
"""
#calculate average error per flux bin
flux_bins = 20; min_count=5
hists = np.empty((len(sdss.xcols), flux_bins))
edges = np.empty((len(sdss.xcols), flux_bins+1))
errors = np.empty((len(sdss.xcols), flux_bins))
remain = np.empty((len(sdss.xcols), flux_bins))
for i, col in enumerate(sdss.xcols):
    hists[i], edges[i] = np.histogram(np.log10(sdss.data[col]), bins=flux_bins)
    
    remain[i] = (hists[i] > min_count)
    print("Number of remaining bins for " + col + ":", len(np.where(remain)[0]))
    #new_edges[i][~(np.insert(remain, len(hists[i]), False) | np.insert(remain, 0, False))] = np.nan
    
    #remaining_edges = edges[i][np.insert(remain, len(hists[i]), False) | np.insert(remain, 0, False)]
    #lower_edges = edges[i][np.insert(remain, len(hists[i]), False)]
    #upper_edges = edges[i][np.insert(remain, 0, False)]

    for j in range(flux_bins):
        if remain[i,j]:
            mask = (np.log10(sdss.data[col]) >= edges[i][j]) & (np.log10(sdss.data[col]) < edges[i][j+1])
            errors[i,j] = np.average(sdss.data['e_'+col][mask])
        else:
            errors[i,j] = np.nan
            
for i, col in enumerate(eagle.xcols):
    print(col)
    for j in range(flux_bins):
        mask = (eagle.data[col] >= 10.**edges[i][j]) & (eagle.data[col] < 10.**edges[i][j+1])
        if remain[i,j]:
            eagle.data[col][mask] = np.random.normal(eagle.data[col][mask], errors[i,j]/2)
        else:
            eagle.data[col][mask] = np.nan
            
            
#calculate average error per flux bin
flux_bins = 20; flux_count=25
errors = np.empty((len(sdss.xcols), flux_bins))
edges = np.empty((len(sdss.xcols), flux_bins+1))
for i, col in enumerate(sdss.xcols):
    hist_sdss, edges[i] = np.histogram(np.log10(sdss.data[col]), bins=flux_bins)
    hist_eagle = np.histogram(np.log10(eagle.data[eagle.xcols[i]]), bins=edges[i])[0]
    remain = (hist_sdss > flux_count) & (hist_eagle > flux_count)
    print("Number of remaining bins for " + col + ":", len(np.where(remain)[0]))

    #if input_uniform and i == 4:
    #    sdss_equal = copy(sdss.data)
    #    eagle_equal = copy(eagle.data)
    #    sdss_equal[:] = np.nan
    #    eagle_equal[:] = np.nan

    for j in range(flux_bins):
        mask_eagle = (eagle.data[eagle.xcols[i]] >= 10.**edges[i,j]) & (eagle.data[eagle.xcols[i]] < 10.**edges[i,j+1])
        mask_sdss = (np.log10(sdss.data[col]) >= edges[i,j]) & (np.log10(sdss.data[col]) < edges[i,j+1])
        if remain[j]:
            if input_variance:
                error = np.average(sdss.data['e_'+col][mask_sdss])
                eagle.data[eagle.xcols[i]][mask_eagle] = np.random.normal(eagle.data[eagle.xcols[i]][mask_eagle], error/2)
            #if input_uniform and i == 4:
            #    perm_sdss = np.random.permutation(hist_sdss[j])[:flux_count]
            #    sdss_equal[j*flux_count:(j+1)*flux_count] = sdss.data[mask_sdss][perm_sdss]
            #    perm_eagle = np.random.permutation(hist_eagle[j])[:flux_count]
            #    eagle_equal[j*flux_count:(j+1)*flux_count] = eagle.data[mask_eagle][perm_eagle]
        else:
            eagle.data[eagle.xcols[i]][mask_eagle] = np.nan
            sdss.data[col][mask_sdss] = np.nan

#if input_uniform:
#    eagle.data = eagle_equal
#    sdss.data = sdss_equal

"""

