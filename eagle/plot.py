import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import cm
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation, ArtistAnimation, writers
from scipy.stats import gaussian_kde
from copy import copy
import shap

rc('text', usetex=True)
plt.style.use('seaborn-colorblind')

plt.rcParams.update({"text.usetex": True, "text.latex.preamble":[r"\usepackage{amsmath}",r"\usepackage{siunitx}",],})

legend_fontsize = 10
title_fontsize = 9

class PLOT_DATA:
    def __init__(self, datanames, **kwargs):
        self.dir = '/disks/strw9/vanweenen/mrp2/plots/'
        self.datanames = datanames
        self.kwargs = kwargs

    def title(self, catalogue):
        title = ''
        for cat in catalogue:
            title += ' ' + cat
        self.savetitle = title; self.plottitle = title
        for count, i in enumerate(self.kwargs):
            self.savetitle += '_' + str(i) + '=' + str(self.kwargs[i])
            self.plottitle += ' $\mid$ ' + str(i) + '=' + str(self.kwargs[i])

    def hist_data(self, catalogue, data, bins_y=20, bins_x=15*[10], xlim=[-6, -2], ylim=[7,12]):
        data_shape = data[0].shape[1]
        row_options = (1,1,1,1,1,2,1,2,3,2,1,3,1,2,3)
        rows = row_options[data_shape-2]
        cols = (data_shape-1) // rows
        
        self.title(catalogue)
        fig = plt.figure(figsize=(3*cols, rows+2))
        fig.suptitle(self.plottitle, fontsize=title_fontsize)
        
        gs = gridspec.GridSpec(rows, cols+1, figure=fig, width_ratios=cols*[1]+[rows], wspace=0)
        gs.tight_layout(fig)
        
        #plot histogram of input
        for i, name in enumerate(self.datanames[:-1]):
            row = i // cols ; col = (i - row*cols) % cols
            ax = plt.subplot(gs[row, col])
            for j, cat in enumerate(catalogue):
                c = next(ax._get_lines.prop_cycler)['color']
                ax.hist(data[j][:,i], bins=bins_x[i], histtype='step', density=True, label=cat, color=c)#, alpha=.5)
                ax.axvline(np.median(data[j][:,i]), linestyle='dashed', linewidth=1, color=c)
            ax.set_xlabel(name)
            ax.set_xlim(xlim)
        
        #share axes and remove ticklabels
        for i, ax in enumerate(fig.axes):
            row = i // cols ; col = (i - row*cols) % cols
            if i != len(self.datanames) - 1:
                if col == 0 and row != rows - 1:
                    ax.get_shared_x_axes().join(ax, fig.axes[row*cols+col])
                    plt.setp(ax.get_xticklabels(), visible=False)
                elif col != 0 and row == rows - 1:
                    ax.get_shared_y_axes().join(ax, fig.axes[row*cols])
                    plt.setp(ax.get_yticklabels(), visible=False)
                elif col != 0 and row != rows - 1:
                    ax.get_shared_x_axes().join(ax, fig.axes[row*cols+col])
                    ax.get_shared_y_axes().join(ax, fig.axes[row*cols])
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
        #plot histogram of output
        ax = plt.subplot(gs[:,cols])
        ax.yaxis.tick_right()
        for j, cat in enumerate(catalogue):
            c = next(ax._get_lines.prop_cycler)['color']
            ax.hist(data[j][:,-1], bins=bins_y, histtype='step', density=True, label=cat, color=c)#, alpha=.5)
            ax.axvline(np.median(data[j][:,-1]), linestyle='dashed', linewidth=1, color=c)
        ax.set_xlabel(self.datanames[-1])
        ax.set_xlim(ylim)
        
        adjust = 0.2 if rows == 1 else 0.1
        plt.subplots_adjust(bottom = adjust)
        
        plt.legend()
        plt.savefig(self.dir + self.savetitle + '.pdf')
        
    def statistical_matrix(self, x, y, catalogue=['eagle'], simple=False, cmap='PiYG'):
        self.title(catalogue)

        C = np.corrcoef(np.hstack((x,y)).T)
        cbar_title = r'Pearson correlation $\rho $'
        
        if not simple:    
            Cm = copy(C)
            C[C < 0.] = np.nan
            Cm[Cm > 0.] = np.nan
            
            vmin = np.trunc(min(np.nanmin(C), -np.nanmax(Cm))*10)/10
            vmax = np.trunc(max(np.nanmax(C), -np.nanmin(Cm))*10)/10
            cmap = 'Greens' ; nticks = 4
        else:
            vmin = np.amin(C) ; vmax = np.amax(C)
            nticks = 8
            
        fig, ax = plt.subplots()
        fig.suptitle('Correlation matrix' + self.plottitle, fontsize=title_fontsize)
        im = ax.imshow(C, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(np.arange(len(self.datanames))) ; ax.set_xticklabels(self.datanames, fontsize=9)
        ax.set_yticks(np.arange(len(self.datanames))) ; ax.set_yticklabels(self.datanames, fontsize=9)
        cbar = fig.colorbar(im, ax=ax) ; cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(nticks))
        if not simple:
            im_m = ax.imshow(Cm, cmap='Reds_r', vmin=-vmax, vmax=-vmin, aspect='auto')
            cbar_m = fig.colorbar(im_m, ax=ax) ; cbar_m.ax.yaxis.set_major_locator(plt.MaxNLocator(nticks))
        cbar.ax.set_title(cbar_title)
        plt.savefig(self.dir + 'Correlation ' + self.savetitle + '.pdf')

class PLOT_NN:
    def __init__(self, io, nn, xnames, ynames, dataname, N, xcols_string, inp, sampling, score = None, epochs = None):
        self.dir = '/disks/strw9/vanweenen/mrp2/plots/'
        self.inp = io.cat
        self.outp = io.ycols[0]
        self.xnames = xnames
        self.ynames = ynames
        self.dataname = dataname
        self.N = N
        self.xcols_string = xcols_string
        self.inp = inp
        self.sampling = sampling
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
        title = 'MLP ' + self.dataname
        self.legendtitle = r'\begin{align*} \text{MAE} &= \num{%.2e} \\ R^2 &= %.3g \\ \bar{R}^2 &= %.3g \\ \mu &= \num{%.2e} \\ \sigma^2 &= \num{%.2e} \end{align*}'%tuple(self.score)
        self.plottitle = title + ' %s $\mid$ %s $\mid$ snap = %s $\mid$ N = %s $\mid$ input = '%(io.sim, io.cat, io.snap, self.N)+self.inp+' $\mid$ sampling = '+self.sampling+' \n'
        self.savetitle = self.dir + title + '_%s_%s_%s_N=%s_input='%(io.sim, io.cat, io.snap, self.N)+self.xcols_string+'_'+self.inp+'_sampling='+self.sampling
        for count, i in enumerate(kwargs):
            self.savetitle += '_' + str(i) + '=' + str(kwargs[i])
            self.plottitle += str(i) + ' = ' + str(kwargs[i])
            if (count%4 == 3 and count != len(kwargs) -1):
                self.plottitle += ' \n '
            else:
                self.plottitle += ' $\mid$ '

    def plot_learning_curve(self, history):
        val_nr = int(len(history.history)/2)
        vals = ('', 'val_', 'val2_', 'val3_', 'val4_')
        val_names = ('train', 'test', 'sdss', 'centrals', 'satellites')
        colors = (self.blue, self.greenblue, self.orange, self.pink, self.green)
        fig, ax = plt.subplots(2,1, squeeze=True, sharex=True)
        plt.suptitle(self.plottitle, fontsize=title_fontsize)
        for i in range(val_nr):
            ax[0].plot(history.epoch, history.history[vals[i]+'loss'], label=val_names[i], color=colors[i])
            ax[1].plot(history.epoch, history.history[vals[i]+'R_squared'], color=colors[i])
        ax[0].set_ylabel('MAE')   ; ax[0].set_xlim((0., np.amax(history.epoch)))
        ax[1].set_ylabel('$R^2$') ; ax[1].set_xlim((0., np.amax(history.epoch)))
        ax[1].set_xlabel('epoch')
        ax[0].legend()
        plt.savefig(self.savetitle+'_learning_rate.pdf')

    def plot_input_output(self, x, y, y_pred, plottype='scatter'):
        if x.shape[1] != 1:
            fig, ax = plt.subplots(y.shape[1],x.shape[1], figsize=(4*x.shape[1],4*y.shape[1]), squeeze=True, sharey=True)
            fig.subplots_adjust(wspace=0, hspace=0)
            ax[0].set_ylabel(self.ynames[0])
            fig.suptitle(self.plottitle, fontsize=title_fontsize-1)
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

    def plot_true_predict(self, y, y_pred, plottype='scatter', cs=False, y_c=None, y_pred_c=None, y_s=None, y_pred_s=None, bins=100):
        label, color, cmap = '', self.greenblue, 'Blues'
        if cs:
            label = 'all'
            clabel, ccolor, ccmap = 'central', self.pink, 'Reds'
            slabel, scolor, scmap = 'satellite', self.green, 'Greens'

        plt.figure(figsize=(6,5))
        plt.title(self.plottitle, fontsize=10)
        plt.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], '--', dashes=(10,10), color='black', lw=.5)

        #scatter
        if plottype == 'scatter':
            plt.scatter(y, y_pred, s=2., marker='o', color=color, edgecolor='none', zorder=1, label=label)
            if cs:
                plt.scatter(y_c, y_pred_c, s=2., marker='o', color=ccolor, edgecolor='none', zorder=1, label=clabel)
                plt.scatter(y_s, y_pred_s, s=2., marker='o', color=scolor, edgecolor='none', zorder=1, label=slabel)
        #hist2d
        elif plottype == 'hist2d':
            h, xedges, yedges, im = plt.hist2d(y, y_pred, bins=bins, cmap=ccmap, cmin=1)
        #hexbin
        elif plottype == 'hexbin':
            plt.hexbin(y, y_pred, gridsize=(bins, bins), cmap=ccmap, mincnt=1)
        #contour(f)
        elif plottype == 'contour' or plottype == 'contourf':
            levels = [0.5, 1., 2., 4., 6.] ; lw = 0.7
            Z, xx, yy, k = kde_fit(y, y_pred) ; cmap_contour = cmap
            if cs:
                Z_c, xx_c, yy_c, k_c = kde_fit(y_c, y_pred_c)
                Z_s, xx_s, yy_s, k_s = kde_fit(y_s, y_pred_s)
                ccmap_contour = ccmap ; scmap_contour = scmap 
            if plottype == 'contourf':
                cmap_contour+='_r'
                CS = plt.contourf(xx, yy, Z, cmap=cmap, cmin=1, levels=levels+[Z.max()], alpha=0.5)
                if cs:
                    ccmap_contour+='_r' ; scmap_contour+='_r'
                    CS_c = plt.contourf(xx_c, yy_c, Z_c, cmap=ccmap, cmin=1, levels=levels+[Z_c.max()], alpha=0.5)
                    CS_s = plt.contourf(xx_s, yy_s, Z_s, cmap=scmap, cmin=1, levels=levels+[Z_s.max()], alpha=0.5)
            elif plottype == 'contour':
                CS = plt.contour(xx, yy, Z, cmap=cmap_contour, cmin=1, levels=levels, linewidths=lw)
                if cs:
                    CS_c = plt.contour(xx_c, yy_c, Z_c, cmap=ccmap_contour, cmin=1, levels=levels, linewidths=lw)
                    CS_s = plt.contour(xx_s, yy_s, Z_s, cmap=scmap_contour, cmin=1, levels=levels, linewidths=lw)
            CS.collections[-2].set_label(label)
            if cs:
                CS_c.collections[-2].set_label(clabel)
                CS_s.collections[-2].set_label(slabel)
        plt.xlim((np.min(y), np.max(y)))
        plt.ylim((np.min(y), np.max(y)))
        plt.xlabel('true ' + self.ynames[0])
        plt.ylabel('predicted ' + self.ynames[0])
        lg = plt.legend(title=self.legendtitle)
        lg.get_title().set_fontsize(10)
        plt.savefig(self.savetitle+'_'+self.outp+'_test-predict'+'_'+plottype+'.pdf')

    def plot_output_error(self, y, y_pred, plottype='scatter', cs=False, y_c=None, y_pred_c=None, y_s=None, y_pred_s=None, bins=20, ylim=(-.23, .23)):
        y = y.reshape(len(y),) ; y_pred = y_pred.reshape(len(y_pred),)
        label, color, cmap = '', self.greenblue, 'Blues'
        if cs:
            label = 'all'
            clabel, ccolor, ccmap = 'central', self.pink, 'Reds'
            slabel, scolor, scmap = 'satellite', self.green, 'Greens'
            y_c = y_c.reshape(len(y_c),) ; y_pred_c = y_pred_c.reshape(len(y_pred_c),)
            y_s = y_s.reshape(len(y_s),) ; y_pred_s = y_pred_s.reshape(len(y_pred_s),)

        plt.figure(figsize=(6,5))
        plt.title(self.plottitle, fontsize=10)
        plt.plot([np.min(y), np.max(y)], [0.,0.], '--', dashes=(10,10), color='black', lw=.5)
        #scatter
        if plottype == 'scatter':
            plt.scatter(y, y_pred - y, s=2., color=color, edgecolor='none', label=label)
            if cs:
                plt.scatter(y_c, y_pred_c - y_c, s=2., color=ccolor, edgecolor='none', label=clabel)
                plt.scatter(y_s, y_pred_s - y_s, s=2., color=scolor, edgecolor='none', label=slabel)
        #hexbin
        elif plottype == 'hexbin':
            plt.hexbin(y, y_pred - y, gridsize=(bins, bins), cmap='Blues', mincnt=1)
        #contour 
        if plottype == 'contour' or plottype == 'contourf':
            cmap_contour = cmap; lw = 0.4 ; levels = [.5, 1., 2., 4., 6.]
            Z, xx, yy, k = kde_fit(y, y_pred-y)
            if plottype == 'contour':
                CS = plt.contour(xx, yy, Z, cmap=cmap_contour, cmin=1, linewidths=lw, levels=levels)
            elif plottype == 'contourf':
                if Z.max() > levels[-1]: levels += [Z.max()]
                CS = plt.contourf(xx, yy, Z, cmap=cmap_contour, cmin=1, levels=levels)
            if cs:
                ccmap_contour = ccmap ; scmap_contour = scmap
                Z_c, xx_c, yy_c, k_c = kde_fit(y_c, y_pred_c-y_c)
                Z_s, xx_s, yy_s, k_s = kde_fit(y_s, y_pred_s-y_s)
                if plottype == 'contour':
                    CS_c = plt.contour(xx_c, yy_c, Z_c, cmap=ccmap_contour, cmin=1, linewidths=lw, levels=levels)
                    CS_s = plt.contour(xx_s, yy_s, Z_s, cmap=scmap_contour, cmin=1, linewidths=lw, levels=levels)
                elif plottype == 'contourf':
                    ccmap_contour+='_r' ; scmap_contour+='_r'
                    if Z.max() > levels[-1]: levels += [Z.max()]
                    CS_c = plt.contourf(xx_c, yy_c, Z_c, cmap=ccmap_contour, cmin=1, alpha=0.5, levels=levels)
                    CS_s = plt.contourf(xx_s, yy_s, Z_s, cmap=scmap_contour, cmin=1, alpha=0.5, levels=levels)
                CS.collections[-2].set_label(label)
                CS_c.collections[-2].set_label(clabel)
                CS_s.collections[-2].set_label(slabel)                
        plt.xlabel('true ' + self.ynames[0])
        plt.ylabel('predicted ' + self.ynames[0] + ' - true ' + self.ynames[0])
        plt.ylim(ylim)
        lg = plt.legend(title=self.legendtitle)
        lg.get_title().set_fontsize(10)
        plt.savefig(self.savetitle+'_'+self.outp+'_error_'+plottype+'.pdf')

    def gif_input_output(self, x, y, Y_pred, mse, r2):
        Writer = writers['ffmpeg'] 
        fig, ax = plt.subplots(1,5, figsize=(20,4), squeeze=True, sharey=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        ax[0].set_ylabel(self.ynames[0])
        fig.suptitle(self.plottitle, fontsize=title_fontsize)

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


class PLOT_SHAP:
    def __init__(self, xnames, xcols_string, inp, sampling):
        self.dir = '/disks/strw9/vanweenen/mrp2/plots/'
        self.xnames = xnames
        self.xcols_string = xcols_string
        self.inp = inp
        self.sampling = sampling
        self.red = shap.plots.colors.red_rgb
        self.blue = shap.plots.colors.blue_rgb
        self.red_blue = shap.plots.colors.red_blue
        self.axis_color = '#333333'
        self.hdots_color = '#cccccc'
        self.vline_color = '#999999'
        self.title()
        
    def title(self):
        self.plottitle = 'SHAP summary input = '+self.inp+' $\mid$ sampling = '+self.sampling
        self.savetitle = self.dir + 'shap_summary_'+self.xcols_string+'_input='+self.inp+'_sampling='+self.sampling

    def summary_plot(self, X, shap_values, dataname='eagle', shap_values_ref=None, dataname_ref='sdss', plottype='bar', dotsize=5):
        feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_range = np.arange(len(self.xnames))
        
        plt.figure()
        plt.title(self.plottitle)
        if plottype == 'bar':
            width = 0.35
            tick_pos = feature_range + width/2
            xlabel = 'mean $\mid \phi_i \mid$ (average impact on model output magnitude)'
            mean_shap = np.abs(shap_values).mean(0)[feature_order]
            mean_shap_ref = np.abs(shap_values_ref).mean(0)[feature_order]
            plt.barh(feature_range, mean_shap, width, label=dataname, color = self.red)
            plt.barh(feature_range+width, mean_shap_ref, width, label=dataname_ref, color = self.blue)
            plt.legend(loc='lower right')
        
        if plottype == 'dot':
            plt.axvline(x=0, color=self.vline_color, linewidth=0.2, zorder=-1)
            for pos, i in enumerate(feature_order):
                tick_pos = feature_range
                xlabel = '$ \phi_i $ (impact on model output magnitude)'
                plt.axhline(y=pos, color=self.hdots_color, lw=1., dashes=(1, 5), zorder=-1)
                shaps = shap_values[:,i]
                values = X[:,i]
                N = len(shaps) ; nbins = 100
                perm = np.random.permutation(N) ; shaps = shaps[perm] ; values = values[perm]
                quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
                inds = np.argsort(quant + np.random.randn(N) * 1e-6)
                layer = 0 ; last_bin = -1 ; row_height=0.4
                ys = np.zeros(N)
                for ind in inds:
                    if quant[ind] != last_bin:
                        layer = 0
                    ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                    layer += 1
                    last_bin = quant[ind]
                ys *= 0.9 * (row_height / np.max(ys + 1))

                #crop vmin and vmax
                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)

                # plot the non-nan values colored by the trimmed feature value
                cvals = values.astype(np.float64)
                cvals_imp = cvals.copy()
                cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                cvals[cvals_imp > vmax] = vmax
                cvals[cvals_imp < vmin] = vmin
                plt.scatter(shaps, pos+ys, cmap=self.red_blue, vmin=vmin, vmax=vmax, s=dotsize, c=cvals, linewidth=0, zorder=3)
            lim = np.amax(np.abs(shap_values)) + 0.1
            plt.xlim((-lim, lim))
            plt.gca().xaxis.set_ticks_position('bottom')
            plt.gca().yaxis.set_ticks_position('none')

            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            #plt.gca().tick_params(color=self.axis_color, labelcolor=self.axis_color)

            #colorbar
            m = cm.ScalarMappable(cmap=self.red_blue)
            m.set_array([0, 1])
            cbar = plt.colorbar(m, ticks=[0, 1], aspect=100)
            cbar.set_ticklabels(['low', 'high'])
            cbar.set_label('Feature value', size=12, labelpad=0)
            cbar.ax.tick_params(labelsize=11, length=0)
            cbar.outline.set_visible(False)

            
        plt.yticks(tick_pos, fontsize=10)
        plt.gca().set_yticklabels([self.xnames[i] for i in feature_order])
        plt.xlabel(xlabel)
        plt.savefig(self.savetitle +'_'+ plottype + '.pdf')

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
