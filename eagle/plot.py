import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation, ArtistAnimation, writers
rc('text', usetex=True)
plt.style.use('seaborn-colorblind')


dir = '/disks/strw9/vanweenen/mrp2/plots/'

def title(sim, cat, snapshot, score=None, MLtype=None, small=False, **kwargs):
    title = '' ; legendtitle = None
    if MLtype != None:
        title = MLtype
        legendtitle = 'MSE = %s, $R^2$ = %s'%(np.around(score[0],3), np.around(score[1],3))
    plottitle = title + ' %s $\mid$ %s $\mid$ snapshot = %s \n'%(sim, cat, snapshot)
    savetitle = title + '_%s_%s_%s'%(sim, cat, snapshot)
    for i in kwargs:
        savetitle += '_' + str(i) + '=' + str(kwargs[i])
        plottitle += str(i) + ' = ' + str(kwargs[i]) + ' $\mid$ '
        if (small and count == 2):
            plottitle += '\n'
    return plottitle, savetitle, legendtitle

def plot_learning_curve(history, epochs, test='validation'):
    fig, ax = plt.subplots(2,1, squeeze=True, sharex=True)
    plt.suptitle('Learning rate')
    ax[0].plot(np.arange(epochs), history.history['loss'], color='grey', label='train')
    ax[0].plot(np.arange(epochs), history.history['val_loss'], color='red', label=test)
    ax[0].set_ylabel('MSE')
    ax[0].set_xlim((0., epochs))
    ax[1].plot(np.arange(epochs), history.history['coeff_determination'], color='grey')
    ax[1].plot(np.arange(epochs), history.history['val_coeff_determination'], color='red')
    ax[1].set_ylabel('$R^2$')
    ax[1].set_xlabel('epoch')
    ax[1].set_xlim((0., epochs))
    ax[0].legend()
    plt.savefig(dir+'Learning_rate_epochs%s.pdf'%epochs)
    #plt.show()

def plot_sed_mstar(x, y, dataset, sim, cat, snapshot, xnames, ynames, predict=True, y_pred=None, score=None, MLtype=None, **kwargs):
    plottitle, savetitle, legendtitle = title(sim, cat, snapshot, score, MLtype, small=False, **kwargs)
    savetitle += '_sed-mstar.pdf'

    fig, ax = plt.subplots(1,5, figsize=(20,4), squeeze=True, sharey=True)
    fig.subplots_adjust(wspace=0, hspace=0)
    ax[0].set_ylabel(ynames[0])
    fig.suptitle(plottitle)
    for i, xname in enumerate(xnames):
        ax[i].scatter(x[:,i], y, s=2., marker='o', label=dataset, color='dimgrey', edgecolor='none')
        if predict:        
            idx = np.argsort(x[:,i])
            x_sort = x[:,i][idx]
            y_sort = y_pred[idx]
            ax[i].scatter(x_sort, y_sort, marker='o', s=2., label='predict', color='red', edgecolor='none')      
        ax[i].set_xlabel(xname)
    plt.legend(title=legendtitle)
    plt.savefig(dir+savetitle)
    #plt.show()


def plot_prediction_test(y, y_pred, sim, cat, snapshot, score=None, MLtype='', **kwargs):
    xname='log $M_{*}$'; yname='predicted log $M_{*}$'
    plottitle, savetitle, legendtitle = title(sim, cat, snapshot, score, MLtype, small=False, **kwargs)
    savetitle += '_test-predict.pdf'
    
    plt.figure(figsize=(8,6))
    plt.title(plottitle)
    plt.plot([np.min(y), np.max(y)], [np.min(y_pred), np.max(y_pred)], color='black', linewidth=.5)
    plt.scatter(y, y_pred, s=2., marker='o', color='green', edgecolor='none', zorder=1)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.legend(title=legendtitle)
    plt.savefig(dir+savetitle)
    #plt.show()
       
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
        ax[i].plot(x[:,i], y, 'ro', markersize=.3, markerfacecolor='dimgrey', markeredgecolor='none', label=dataset)        
        ax[i].set_xlabel(xname)

        idx = np.argsort(x[:,i])
        x_sort = x[:,i][idx]
        y_sort = Y_pred[0][idx]
        scatter, = ax[i].plot(x_sort, y_sort, 'ro', markersize=.3, markerfacecolor='red', markeredgecolor='none', label='predict')
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
        plt.legend(title= 'MSE = %s, $R^2$ = %s'%(np.around(mse[n],3), np.around(r2[n],3))+ ' epoch %s'%n)
        return scatters

    anim = FuncAnimation(fig, update, fargs = (scatters, x, Y_pred, mse, r2), frames = np.arange(kwargs['epochs']), save_count=20, interval=1000)
    anim.save(dir+savetitle, dpi=300, writer=Writer(fps=1, metadata=dict(artist='Eva'), bitrate=1800))
    plt.show()
