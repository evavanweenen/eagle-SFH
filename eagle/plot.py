import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation, ArtistAnimation, writers
from scipy.stats import gaussian_kde

rc('text', usetex=True)
plt.style.use('seaborn-colorblind')

plt.rcParams.update({"text.usetex": True, "text.latex.preamble":[r"\usepackage{amsmath}",r"\usepackage{siunitx}",],})

legend_fontsize = 10
title_fontsize = 10

class PLOT_DATA:
    def __init__(self, datanames, **kwargs):
        self.dir = '/disks/strw9/vanweenen/mrp2/plots/'
        self.datanames = datanames
        self.kwargs = kwargs

    def title(self, catalogue):
        title = 'Data '
        for cat in catalogue:
            title += ' ' + cat
        self.savetitle = title; self.plottitle = title
        for count, i in enumerate(self.kwargs):
            self.savetitle += '_' + str(i) + '=' + str(self.kwargs[i])
            self.plottitle += ' $\mid$ ' + str(i) + '=' + str(self.kwargs[i])

    def hist_data(self, catalogue, data, bins=20, xlim=[-6, -2], ylim=[7,12]):
        self.title(catalogue)
        fig, ax = plt.subplots(1,data[0].shape[1], figsize=(4*data[0].shape[1],4), squeeze=True, sharey=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.suptitle(self.plottitle)
        for i, name in enumerate(self.datanames):
            for j, cat in enumerate(catalogue):
                ax[i].hist(data[j][:,i], bins=bins, histtype='step', density=True, label=cat, alpha=.5)
            ax[i].set_xlabel(name)
            if i != 5:
                ax[i].set_xlim(xlim)
            else:
                ax[i].set_xlim(ylim)
        plt.legend()
        plt.savefig(self.dir + self.savetitle + '.pdf')

class PLOT_NN:
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
        self.pink = '#D300B0'
        self.green = '#77AA00' #83D300' #A9F200
        self.greenblue = '#1F91BF'#008080'#00A470'

        self.title(io, ep=self.epochs, hnodes=list(filter(None, nn.h_nodes)), act=list(filter(None, nn.activation)), drop=list(filter(None, nn.dropout)), optim=nn.optimizer, b=nn.batch_size)

    def title(self, io, **kwargs):
        title = self.MLtype
        self.legendtitle = r'\begin{align*} \text{MSE} &= \num{%.2e} \\ R^2 &= %.3g \\ \bar{R}^2 &= %.3g \\ \mu &= \num{%.2e} \\ \sigma &= \num{%.2e} \end{align*}'%tuple(self.score)
        self.plottitle = title + ' %s $\mid$ %s $\mid$ snapshot = %s \n'%(io.sim, io.cat, io.snap)
        self.savetitle = self.dir + title + '_%s_%s_%s'%(io.sim, io.cat, io.snap)
        for count, i in enumerate(kwargs):
            self.savetitle += '_' + str(i) + '=' + str(kwargs[i])
            self.plottitle += str(i) + ' = ' + str(kwargs[i])
            if (count%3 == 2 and count != len(kwargs) -1):
                self.plottitle += ' \n '
            else:
                self.plottitle += ' $\mid$ '

    def plot_learning_curve(self, history, epochs):
        fig, ax = plt.subplots(2,1, squeeze=True, sharex=True)
        plt.suptitle(self.plottitle, fontsize=10)
        ax[0].plot(np.arange(epochs), history.history['loss'], color=self.blue, label='train')
        ax[0].plot(np.arange(epochs), history.history['val_loss'], color=self.orange, label='test')
        ax[0].set_ylabel('MSE')
        ax[0].set_xlim((0., epochs))
        ax[1].plot(np.arange(epochs), history.history['R_squared'], color=self.blue)
        ax[1].plot(np.arange(epochs), history.history['val_R_squared'], color=self.orange)
        ax[1].set_ylabel('$R^2$')
        ax[1].set_xlabel('epoch')
        ax[1].set_xlim((0., epochs))
        ax[0].legend()
        plt.savefig(self.savetitle+'_learning_rate.pdf')

    def plot_input_output(self, x, y, y_pred, plottype='scatter'):
        if x.shape[1] != 1:
            fig, ax = plt.subplots(y.shape[1],x.shape[1], figsize=(4*x.shape[1],4*y.shape[1]), squeeze=True, sharey=True)
            fig.subplots_adjust(wspace=0, hspace=0)
            ax[0].set_ylabel(self.ynames[0])
            fig.suptitle(self.plottitle, fontsize=8)
            for i, xname in enumerate(self.xnames):
                idx = np.argsort(x[:,i])
                x_sort = x[:,i][idx]
                y_sort = y_pred[idx]
                if plottype == 'scatter':
                    ax[i].plot(x[:,i], y, 'o', markersize=1., markerfacecolor=self.blue, mec='none', label='true')
                    ax[i].plot(x_sort, y_sort, 'o', markersize=1., markerfacecolor=self.orange, mec='none', label='predict')
                elif plottype == 'contour':
                    cmap = 'Blues' ; cmap_p = 'Oranges'; levels = [0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.]
                    Z, xx, yy, k = kde_fit(x[:,i], y)
                    Z_p, xx_p, yy_p, k_p = kde_fit(x_sort, y_sort)
                    #CS = ax[i].contourf(xx, yy, Z, levels=levels+[Z.max()], cmap=cmap, cmin=1, alpha=0.5)
                    #CS_p = ax[i].contourf(xx_p, yy_p, Z_p, levels=levels+[Z_p.max()], cmap=cmap_p, cmin=1, alpha=0.5)
                    CS = ax[i].contour(xx, yy, Z, cmap=cmap, levels=levels, linewidths=0.4)
                    CS_p = ax[i].contour(xx_p, yy_p, Z_p, cmap=cmap_p, levels=levels, linewidths=0.4)
                ax[i].set_xlabel(xname)
        else:
            fig, ax = plt.subplots(y.shape[1],x.shape[1], figsize=(5*x.shape[1],4*y.shape[1]))         
            ax.set_ylabel(self.ynames[0])
            fig.suptitle(self.plottitle, fontsize=8)
            idx = np.argsort(x[:,0])
            x_sort = x[:,0][idx]
            y_sort = y_pred[idx]
            if plottype == 'scatter':
                ax.plot(x, y, 'o', markersize=1., markerfacecolor=self.blue, mec='none', label='true')
                ax.plot(x_sort, y_sort, 'o', markersize=1., markerfacecolor=self.orange, mec='none', label='predict')
            elif plottype == 'contour':
                cmap = 'Blues' ; cmap_p = 'Oranges' ; levels = [0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.]
                Z, xx, yy, k = kde_fit(x, y)
                Z_p, xx_p, yy_p, k_p = kde_fit(x_sort, y_sort)
                #CS = ax.contourf(xx, yy, Z, cmap=cmap, cmin=1, levels=levels+[Z.max()], alpha=0.5)
                #CS_p = ax.contourf(xx_p, yy_p, Z_p, cmap=cmap_p, cmin=1, levels=levels+[Z_p.max()], alpha=0.5)
                CS = ax.contour(xx, yy, Z, cmap=cmap, linewidths=0.4)
                CS_p = ax.contour(xx_p, yy_p, Z_p, cmap=cmap_p, linewidths=0.4)
            ax.set_xlabel(self.xnames[0])
        if plottype == 'contour':
            CS.collections[-1].set_label('true') ; CS_p.collections[-1].set_label('predict')
        lg = plt.legend(title=self.legendtitle)
        lg.get_title().set_fontsize(10)
        plt.savefig(self.savetitle + '_'+self.inp+'-'+self.outp+'_'+plottype+'.pdf')

    def plot_true_predict(self, y, y_pred, plottype='scatter', cs_mask=None, bins=100):
        if cs_mask is not None:
            clabel, ccolor, ccmap = 'central', self.pink, 'Reds'
            slabel, scolor, scmap = 'satellite', self.green, 'Greens'
        else:
            cs_mask = np.full(y.shape, True)
            clabel, ccolor, ccmap = '', self.greenblue, 'Blues'
            slabel, scolor, scmap = None, None, None

        plt.figure(figsize=(6,5))
        plt.title(self.plottitle, fontsize=10)
        plt.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], '--', dashes=(10,10), color='black', lw=.5)

        #scatter
        if plottype == 'scatter':
            plt.scatter(y[cs_mask], y_pred[cs_mask], s=2., marker='o', color=ccolor, edgecolor='none', zorder=1, label=clabel)
            if not cs_mask.all():
                plt.scatter(y[~cs_mask], y_pred[~cs_mask], s=2., marker='o', color=scolor, edgecolor='none', zorder=1, label=slabel)
        #hist2d
        elif plottype == 'hist2d':
            h, xedges, yedges, im = plt.hist2d(y, y_pred, bins=bins, cmap=ccmap, cmin=1)
        #hexbin
        elif plottype == 'hexbin':
            plt.hexbin(y, y_pred, gridsize=(bins, bins), cmap=ccmap, mincnt=1)
        #contour(f)
        elif plottype == 'contour' or plottype == 'contourf':
            levels = [0.5, 1., 2., 4., 6.]
            Z_c, xx_c, yy_c, k_c = kde_fit(y[cs_mask], y_pred[cs_mask])
            lw = 0.4 ; ccmap_contour = ccmap ; scmap_contour = scmap
            if plottype == 'contourf':
                lw = 0.7 ; ccmap_contour+='_r'
                CS_c = plt.contourf(xx_c, yy_c, Z_c, cmap=ccmap, cmin=1, levels=levels+[Z_c.max()], alpha=0.5)
            elif plottype == 'contour':
                CS_c = plt.contour(xx_c, yy_c, Z_c, cmap=ccmap_contour, cmin=1, levels=levels, linewidths=lw)
            CS_c.collections[-1].set_label(clabel)

            if not cs_mask.all():
                Z_s, xx_s, yy_s, k_s = kde_fit(y[~cs_mask], y_pred[~cs_mask])
                if plottype == 'contourf':
                    scmap_contour+='_r'
                    CS_s = plt.contourf(xx_s, yy_s, Z_s, cmap=scmap, cmin=1, levels=levels+[Z_s.max()], alpha=0.5)
                elif plottype == 'contour':
                    CS_s = plt.contour(xx_s, yy_s, Z_s, cmap=scmap_contour, cmin=1, levels=levels, linewidths=0.7)
                CS_s.collections[-1].set_label(slabel)

        plt.xlabel('true ' + self.ynames[0])
        plt.ylabel('predicted ' + self.ynames[0])
        lg = plt.legend(title=self.legendtitle)
        lg.get_title().set_fontsize(10)
        plt.savefig(self.savetitle+'_'+self.outp+'_test-predict'+'_'+plottype+'.pdf')

    def plot_output_error(self, y, y_pred, plottype='scatter', cs_mask=None, bins=20, ylim=(-.23, .23)):
        y = y.reshape(len(y),) ; y_pred = y_pred.reshape(len(y_pred),)
        if cs_mask is not None:
            clabel, ccolor, ccmap = 'central', self.pink, 'Reds'
            slabel, scolor, scmap = 'satellite', self.green, 'Greens'
        else:
            cs_mask = np.full(y.shape, True)
            clabel, ccolor, ccmap = '', self.greenblue, 'Blues'
            slabel, scolor, scmap = None, None, None

        plt.figure(figsize=(6,5))
        plt.title(self.plottitle, fontsize=10)
        plt.plot([np.min(y), np.max(y)], [0.,0.], '--', dashes=(10,10), color='black', lw=.5)
        if plottype == 'scatter':
            plt.scatter(y[cs_mask], y_pred[cs_mask] - y[cs_mask], s=2., color=ccolor, edgecolor='none', label=clabel)
            if cs_mask is not None:
                plt.scatter(y[~cs_mask], y_pred[~cs_mask] - y[~cs_mask], s=2., color=scolor, edgecolor='none', label=slabel)
        elif plottype == 'hexbin':
            plt.hexbin(y, y_pred - y, gridsize=(bins, bins), cmap='Blues', mincnt=1) 
        plt.xlabel('true ' + self.ynames[0])
        plt.ylabel('predicted ' + self.ynames[0] + ' - true ' + self.ynames[0])
        plt.ylim(ylim)
        if plottype == 'contour':
            levels = [.5, 1., 2., 4., 6.]
            Z_c, xx_c, yy_c, k_c = kde_fit(y[cs_mask], y_pred[cs_mask]-y[cs_mask])
            lw = 0.4 ; ccmap_contour = ccmap ; scmap_contour = scmap
            CS_c = plt.contour(xx_c, yy_c, Z_c, cmap=ccmap_contour, cmin=1, levels=levels, linewidths=lw)
            CS_c.collections[-1].set_label(clabel)
            if not cs_mask.all():
                Z_s, xx_s, yy_s, k_s = kde_fit(y[~cs_mask], y_pred[~cs_mask]-y[~cs_mask])
                CS_s = plt.contour(xx_s, yy_s, Z_s, cmap=scmap_contour, cmin=1, levels=levels, linewidths=0.7)
                CS_s.collections[-1].set_label(slabel)
        lg = plt.legend(title=self.legendtitle, loc='lower right')
        lg.get_title().set_fontsize(10)
        plt.savefig(self.savetitle+'_'+self.outp+'_error_'+plottype+'.pdf')

    def gif_input_output(self, x, y, Y_pred, mse, r2):
        Writer = writers['ffmpeg'] 
        fig, ax = plt.subplots(1,5, figsize=(20,4), squeeze=True, sharey=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        ax[0].set_ylabel(self.ynames[0])
        fig.suptitle(self.plottitle)

        scatters = []
        for i, xname in enumerate(self.xnames):
            ax[i].plot(x[:,i], y, 'o', markersize=1., markerfacecolor=self.blue, mec='none', label='target')
            ax[i].set_xlabel(xname)

            idx = np.argsort(x[:,i])
            x_sort = x[:,i][idx]
            y_sort = Y_pred[0][idx]
            scatter, = ax[i].plot(x_sort, y_sort, 'o', markersize=1., markerfacecolor=self.orange, mec='none', label='predict')
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

def kde_fit(x, y, bins=100):
    x = x.reshape(len(x),) ; y = y.reshape(len(y),)
    xx, yy = np.mgrid[x.min():x.max():bins*1j, y.min():y.max():bins*1j]
    k = gaussian_kde(np.vstack([x, y]))
    Z = np.reshape(k(np.vstack([xx.flatten(), yy.flatten()])).T, xx.shape)
    return Z, xx, yy, k


def plot_gridsearch(score, grid_values, grid_edges, grid_names, name):
    dim = len(grid_names)
    cmap = 'viridis'
    if score == 'mse':
        cmap += '_r'
        lscore = 'MSE'
    else:
        lscore = '$R^2$'
    """
        fig, ax = plt.subplots(2, 1, figsize=(7,6), squeeze=True, sharex=True)
        ax[0].scatter(np.arange(len(grid_edges)), grid_mse)
        ax[1].scatter(np.arange(len(grid_edges)), grid_r2)
        ax[0].set_ylabel('MSE')
        ax[1].set_ylabel('$R^2$')
        ax[1].set_xticks(np.arange(len(grid_edges)))
        ax[1].set_xticklabels(np.array(grid_edges).astype('str').tolist())
    """
    fig, ax = plt.subplots(dim, dim, figsize=(7,6), squeeze=True)
    vmin = np.amin(grid_values) ; vmax = np.amax(grid_values)
    for i in range(dim):
        for j in range(dim):
            axis = tuple([x for x in np.arange(dim) if x!=i and x!=j])
            if i == j:
                diag = np.diagflat(np.average(grid_values, axis=axis))
                diag[diag == 0] = np.nan
                im = ax[i,j].imshow(diag, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
            if i < j:
                im = ax[i,j].imshow(np.average(grid_values, axis=axis), vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
            elif j < i:
                im = ax[i,j].imshow(np.average(grid_values, axis=axis).T, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            ax[dim-1,j].set_xticks(np.arange(len(grid_edges[j])))
            ax[dim-1,j].set_xticklabels(np.array(grid_edges[j]).astype('str').tolist())
            ax[dim-1,j].set_xlabel(grid_names[j])
        ax[i,0].set_ylabel(grid_names[i])
        ax[i,0].set_yticks(np.arange(len(grid_edges[i])))
        ax[i,0].set_yticklabels(np.array(grid_edges[i]).astype('str').tolist())
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_title('average ' + lscore)
    plt.savefig('/disks/strw9/vanweenen/mrp2/plots/'+'GridSearch' + str(grid_names) + name + score + '.pdf')
    plt.show()

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
