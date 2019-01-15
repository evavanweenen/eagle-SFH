import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation, ArtistAnimation, writers
rc('text', usetex=True)
plt.style.use('seaborn-colorblind')

class Plot:
    def __init__(self, io, nn, xnames, ynames, MLtype, score = None, epochs = None):
        self.dir = '/disks/strw9/vanweenen/mrp2/plots/'
        self.inp = io.cat
        self.outp = io.ycols[0]
        self.xnames = xnames
        self.ynames = ynames
        self.MLtype = MLtype
        self.score = score
        if epochs == None:
            self.epochs = nn.epochs
        else:
            self.epochs = epochs
        self.orange = '#FF6500'
        self.blue = '#009DFF'  ##33D6FF  

        self.title(io, ep=self.epochs, hneurons=nn.h_neurons, act=nn.activation_h, drop=nn.dropout, optim=nn.optimizer, b=nn.batch_size)

    def title(self, io, small=False, **kwargs):
        title = self.MLtype
        self.legendtitle = 'MSE = %.2g, $R^2$ = %.2g'%(self.score[0],self.score[1])
        self.plottitle = title + ' %s $\mid$ %s $\mid$ snapshot = %s \n'%(io.sim, io.cat, io.snap)
        self.savetitle = self.dir + title + '_%s_%s_%s'%(io.sim, io.cat, io.snap)
        for count, i in enumerate(kwargs):
            self.savetitle += '_' + str(i) + '=' + str(kwargs[i])
            self.plottitle += str(i) + ' = ' + str(kwargs[i]) + ' $\mid$ '
            #if (small == True and count%3 == 2):
            #    plottitle += ' \n '

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
        plt.savefig(self.dir+'sfh_universe.pdf')

    def plot_learning_curve(self, history, epochs):
        fig, ax = plt.subplots(2,1, squeeze=True, sharex=True)
        plt.suptitle('Learning rate')
        ax[0].plot(np.arange(epochs), history.history['loss'], color=self.blue, label='train')
        ax[0].plot(np.arange(epochs), history.history['val_loss'], color=self.orange, label='test')
        ax[0].set_ylabel('MSE')
        ax[0].set_xlim((0., epochs))
        ax[1].plot(np.arange(epochs), history.history['coeff_determination'], color=self.blue)
        ax[1].plot(np.arange(epochs), history.history['val_coeff_determination'], color=self.orange)
        ax[1].set_ylabel('$R^2$')
        ax[1].set_xlabel('epoch')
        ax[1].set_xlim((0., epochs))
        ax[0].legend()
        plt.savefig(self.savetitle+'+_learning_rate.pdf')

    def plot_input_output(self, x, y, y_pred):
        fig, ax = plt.subplots(y.shape[1],x.shape[1], figsize=(4*x.shape[1],4*y.shape[1]), squeeze=True, sharey=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        ax[0].set_ylabel(self.ynames[0])
        fig.suptitle(self.plottitle)
        for i, xname in enumerate(self.xnames):
            ax[i].plot(x[:,i], y, 'o', markersize=1., markerfacecolor=self.blue, markeredgecolor='none', label='target')       
            idx = np.argsort(x[:,i])
            x_sort = x[:,i][idx]
            y_sort = y_pred[idx]
            ax[i].plot(x_sort, y_sort, 'o', markersize=1., markerfacecolor=self.orange, markeredgecolor='none', label='predict')
            ax[i].set_xlabel(xname)
        plt.legend(title=self.legendtitle)
        plt.savefig(self.savetitle + '_'+self.inp+'-'+self.outp+'.pdf')

    def plot_true_predict(self, y, y_pred):
        plt.figure(figsize=(8,6))
        plt.title(self.plottitle)
        plt.plot([np.min(y), np.max(y)], [np.min(y_pred), np.max(y_pred)], color='black', linewidth=.5)
        plt.scatter(y, y_pred, s=2., marker='o', color='green', edgecolor='none', zorder=1)
        plt.xlabel('target log $M_{*}$')
        plt.ylabel('predicted log $M_{*}$')
        plt.legend(title=self.legendtitle)
        plt.savefig(self.savetitle+'_'+self.outp+'_test-predict.pdf')
           
    def gif_input_output(self, x, y, Y_pred, mse, r2):
        Writer = writers['ffmpeg'] 
        fig, ax = plt.subplots(1,5, figsize=(20,4), squeeze=True, sharey=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        ax[0].set_ylabel(self.ynames[0])
        fig.suptitle(self.plottitle)

        scatters = []
        for i, xname in enumerate(self.xnames):
            ax[i].plot(x[:,i], y, 'o', markersize=1., markerfacecolor=self.blue, markeredgecolor='none', label='target')        
            ax[i].set_xlabel(xname)

            idx = np.argsort(x[:,i])
            x_sort = x[:,i][idx]
            y_sort = Y_pred[0][idx]
            scatter, = ax[i].plot(x_sort, y_sort, 'o', markersize=1., markerfacecolor=self.orange, markeredgecolor='none', label='predict')
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
        anim.save(self.savetitle+'_sed-mstar.mp4', dpi=300, writer=Writer(fps=1, metadata=dict(artist='Eva'), bitrate=1800))
