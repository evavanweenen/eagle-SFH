import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation, ArtistAnimation, writers
rc('text', usetex=True)
plt.style.use('seaborn-colorblind')


dir = '/disks/strw9/vanweenen/mrp2/plots/'

orange = '#FF6500'
blue = '#009DFF'  ##33D6FF  

def title(sim, cat, snapshot, score=None, MLtype=None, small=False, **kwargs):
    title = '' ; legendtitle = None
    if MLtype != None:
        title = MLtype
        legendtitle = 'MSE = %.2g, $R^2$ = %.2g'%(score[0],score[1])
    plottitle = title + ' %s $\mid$ %s $\mid$ snapshot = %s \n'%(sim, cat, snapshot)
    savetitle = title + '_%s_%s_%s'%(sim, cat, snapshot)
    for count, i in enumerate(kwargs):
        savetitle += '_' + str(i) + '=' + str(kwargs[i])
        plottitle += str(i) + ' = ' + str(kwargs[i]) + ' $\mid$ '
        #if (small == True and count%3 == 2):
        #    plottitle += ' \n '
    return plottitle, savetitle, legendtitle

def plot_sfr_z(sims, L, sfrs, zs):
    plt.figure()
    plt.title('Star formation history of the universe')
    for i, sim in enumerate(sims):
        sfr_sum = np.array(sfrs[i].sum(axis=0))[0]
        plt.plot(zs[i], sfr_sum/(L[i]**3), label=sim)
    plt.xscale('log')
    plt.xlim(6*10**-2, 10**1)
    plt.xlabel('$z$')
    plt.ylabel('$\sum_i SFR_i$ ($M_{\odot} yr^{-1} Mpc^{-3}$)')
    plt.legend()
    plt.savefig(dir+'sfh_universe.pdf')

def plot_learning_curve(history, epochs, test='validation'):
    fig, ax = plt.subplots(2,1, squeeze=True, sharex=True)
    plt.suptitle('Learning rate')
    ax[0].plot(np.arange(epochs), history.history['loss'], color=blue, label='train')
    ax[0].plot(np.arange(epochs), history.history['val_loss'], color=orange, label=test)
    ax[0].set_ylabel('MSE')
    ax[0].set_xlim((0., epochs))
    ax[1].plot(np.arange(epochs), history.history['coeff_determination'], color=blue)
    ax[1].plot(np.arange(epochs), history.history['val_coeff_determination'], color=orange)
    ax[1].set_ylabel('$R^2$')
    ax[1].set_xlabel('epoch')
    ax[1].set_xlim((0., epochs))
    ax[0].legend()
    plt.savefig(dir+'Learning_rate_epochs%s.pdf'%epochs)

def plot_input_output(inp, outp, x, y, dataset, sim, cat, snapshot, xnames, ynames, predict=True, y_pred=None, score=None, MLtype=None, **kwargs):
    plottitle, savetitle, legendtitle = title(sim, cat, snapshot, score, MLtype, small=False, **kwargs)
    savetitle += '_'+inp+'-'+outp+'.pdf'

    fig, ax = plt.subplots(1,5, figsize=(20,4), squeeze=True, sharey=True)
    fig.subplots_adjust(wspace=0, hspace=0)
    ax[0].set_ylabel(ynames[0])
    fig.suptitle(plottitle)
    for i, xname in enumerate(xnames):
        ax[i].plot(x[:,i], y, 'o', markersize=1., markerfacecolor=blue, markeredgecolor='none', label='target')
        #ax[i].scatter(x[:,i], y, s=2., marker='o', label='target', color=blue, edgecolor='none')
        if predict:        
            idx = np.argsort(x[:,i])
            x_sort = x[:,i][idx]
            y_sort = y_pred[idx]
            ax[i].plot(x_sort, y_sort, 'o', markersize=1., markerfacecolor=orange, markeredgecolor='none', label='predict')
            #ax[i].scatter(x_sort, y_sort, marker='o', s=2., label='predict', color=orange, edgecolor='none')      
        ax[i].set_xlabel(xname)
    plt.legend(title=legendtitle)
    plt.savefig(dir+savetitle)


def plot_prediction_test(outp, y, y_pred, sim, cat, snapshot, score=None, MLtype='', **kwargs):
    xname='target log $M_{*}$'; yname='predicted log $M_{*}$'
    plottitle, savetitle, legendtitle = title(sim, cat, snapshot, score, MLtype, small=True, **kwargs)
    savetitle += '_'+outp+'_test-predict.pdf'
    
    plt.figure(figsize=(8,6))
    plt.title(plottitle)
    plt.plot([np.min(y), np.max(y)], [np.min(y_pred), np.max(y_pred)], color='black', linewidth=.5)
    plt.scatter(y, y_pred, s=2., marker='o', color='green', edgecolor='none', zorder=1)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.legend(title=legendtitle)
    plt.savefig(dir+savetitle)
       
def gif_sed_mstar(x, y, Y_pred, dataset, sim, cat, snapshot, xnames, ynames, mse, r2, MLtype=None, **kwargs):
    plottitle, savetitle, legendtitle = title(sim, cat, snapshot, [mse[19],r2[19]], MLtype, small=False, **kwargs)
    savetitle += '_sed-mstar.mp4'
    Writer = writers['ffmpeg'] 

    fig, ax = plt.subplots(1,5, figsize=(20,4), squeeze=True, sharey=True)
    fig.subplots_adjust(wspace=0, hspace=0)
    
    ax[0].set_ylabel(ynames[0])
    fig.suptitle(plottitle)

    scatters = []
    for i, xname in enumerate(xnames):
        ax[i].plot(x[:,i], y, 'o', markersize=1., markerfacecolor=blue, markeredgecolor='none', label='target')        
        ax[i].set_xlabel(xname)

        idx = np.argsort(x[:,i])
        x_sort = x[:,i][idx]
        y_sort = Y_pred[0][idx]
        scatter, = ax[i].plot(x_sort, y_sort, 'o', markersize=1., markerfacecolor=orange, markeredgecolor='none', label='predict')
        scatters.append(scatter)
    """
    def init():
        for scatter in scatters:
            scatter.set_offsets([[],[]])
        return scatters
    """
    def update(n, scatters, x, Y_pred, mse, r2):
        for i, scatter in enumerate(scatters):
            idx = np.argsort(x[:,i])
            y_sort = Y_pred[n][idx]
            scatter.set_ydata(y_sort)
        plt.legend(title= 'MSE = %.2g, $R^2$ = %.2g'%(mse[n], r2[n])+ ' epoch %s'%n)
        return scatters

    anim = FuncAnimation(fig, update, fargs = (scatters, x, Y_pred, mse, r2), frames = np.arange(kwargs['epochs']), save_count=20, interval=1000)
    anim.save(dir+savetitle, dpi=300, writer=Writer(fps=1, metadata=dict(artist='Eva'), bitrate=1800))
