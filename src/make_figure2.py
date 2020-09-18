import numpy as np
import matplotlib.pyplot as plt
import plotting_convention as plotting_convention
import LFPy
import neuron
from matplotlib import colorbar
from matplotlib.collections import PolyCollection
import os
from os.path import join

def make_fig_1(totnsegs, imem, syninds, xmids, zmids, zips,
               cb_LFP_close, cb_LFP_far,
               multi_dip_LFP_close, multi_dip_LFP_far,
               db_LFP_close, db_LFP_far,
               LFP_max,
               time_max,
               multi_dips, multi_dip_locs,
               single_dip, r_mid,
               X, Z, X_f, Z_f):

    plt.interactive(1)
    plt.close('all')

    fig = plt.figure()
    ax0 = plt.subplot2grid((3,3),(0,0))
    ax1 = plt.subplot2grid((3,3),(1,0))
    ax2 = plt.subplot2grid((3,3),(2,0))
    ax3 = plt.subplot2grid((3,3),(0,1))
    ax4 = plt.subplot2grid((3,3),(1,1))
    ax5 = plt.subplot2grid((3,3),(2,1))
    ax6 = plt.subplot2grid((3,3),(0,2))
    ax7 = plt.subplot2grid((3,3),(1,2))
    ax8 = plt.subplot2grid((3,3),(2,2))

    # plot neuron morphology in top row
    plot_neuron(ax0, syninds, xmids, zmids, zips, syn=True, lengthbar=True)
    plot_neuron(ax3, syninds, xmids, zmids, zips, syn=True, lengthbar=True, lb_clr='w')
    plot_neuron(ax6, syninds, xmids, zmids, zips, syn=True, lengthbar=True, lb_clr='w')

    # plot transmembrane currents
    for idx in range(totnsegs):
        arrowlength = np.abs(imem[idx, time_max])*1e5
        wdth = 1.
        if [idx] == syninds:
            arrowlength = -700.
            wdth = 2.
            ax0.arrow(xmids[idx]-arrowlength, zmids[idx],
                       arrowlength, 0.,
                       width = 4.,
                       head_length = 39.,
                       head_width = 30.,
                       length_includes_head = True, color='#0D325F',
                    #    alpha=.5
                       )
        else:
            ax0.arrow(xmids[idx], zmids[idx],
                       arrowlength, 0.,
                       width = wdth,
                       head_length = 3.4,
                       head_width = 7.,
                       length_includes_head = True, color='#D90011',
                       alpha=.5)

    # plt lfp close
    plot_lfp(fig, ax1, cb_LFP_close, LFP_max, time_max, X, Z, lengthbar=True)
    plot_lfp(fig, ax4, multi_dip_LFP_close, LFP_max, time_max, X, Z, lengthbar=False)
    plot_lfp(fig, ax7, db_LFP_close, LFP_max, time_max, X, Z, lengthbar=False)

    # plot lfp far
    plot_lfp_far(fig, ax2, cb_LFP_far, LFP_max, time_max, X_f, Z_f, lengthbar=True)
    plot_lfp_far(fig, ax5, multi_dip_LFP_far, LFP_max, time_max, X_f, Z_f, lengthbar=False)
    LFP, ep_intervals, ticks = plot_lfp_far(fig, ax8, db_LFP_far, LFP_max, time_max, X_f, Z_f, lengthbar=False, colorax=True)

    # plot neurons in second row
    for ax in [ax1, ax4]:
        plot_neuron(ax, syninds, xmids, zmids, zips, syn=False, clr='w')

    # plot multi-dipole arrows
    for i in range(len(multi_dips)):
        p = multi_dips[i, time_max]*10
        loc = multi_dip_locs[i]
        w = np.linalg.norm(p)*2
        ax3.arrow(loc[0] - 0.5*p[0],
                  loc[2] - 0.5*p[2],
                  p[0], p[2],
                  color='green', alpha=0.8, width = w, head_width = 4*w,
                  length_includes_head = False
                  )

    # plot single dipole arrow
    arrow = single_dip[time_max]*25

    arrow_colors = ['gray', 'w']
    arrow_axes = [ax6, ax7]
    for i in range(2):
        arrow_axes[i].arrow(r_mid[0] - 2*arrow[0],
                            r_mid[2] - 2*arrow[2],
                            arrow[0]*4, arrow[2]*4, #fc = 'k',ec = 'k',
                            color=arrow_colors[i], alpha=0.8, width = 12, #head_width = 60.,
                            length_includes_head = True)#,



    # colorbar
    cax = fig.add_axes([0.13, 0.03, 0.8, 0.01])
    cbar = fig.colorbar(ep_intervals,cax=cax,
                orientation='horizontal', format='%3.3f',
                extend = 'max')
    cbar.set_ticks(ticks)
    labels = [r'$-10^{\/\/2}$', r'$-10^{\/\/1}$', r'$-1.0$', r'$-10^{-1}$',r'$-10^{-2}$',
              r'$\/\/10^{-2}$', r'$\/\/10^{-1}$', r'$\/\/1.0$',
              r'$\/\/10^{\/\/1}$', r'$\/\/10^{\/\/2}$']
    cax.set_xticklabels(labels, rotation = 40)
    cbar.set_label('$\phi$ (nV)',labelpad = -0.4)
    cbar.outline.set_visible(False)

    ax0.set_title('compartment-based', fontsize='x-small')
    ax3.set_title('multi-dipole', fontsize='x-small')
    ax6.set_title('single-dipole', fontsize='x-small')

    fig.set_size_inches(8, 8)

    plotting_convention.mark_subplots([ax0, ax3, ax6], letters='ABC', xpos=-0.02, ypos=1.0)
    plotting_convention.mark_subplots([ax1, ax4, ax7], letters='DEF', xpos=-0.02, ypos=1.05)
    plotting_convention.mark_subplots([ax2, ax5, ax8], letters='GHI', xpos=-0.02, ypos=0.94)

    for ax in [ax1, ax2, ax4, ax5, ax7, ax8, ax0, ax3, ax6]:
        ax.set_aspect('equal', 'datalim')
    plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.1, bottom=0.05, right=0.96, top=0.93)
    return fig

def plot_lfp(fig, ax, LFP_measurements, max_LFP, timestep, X, Z, colorax = False,
             lengthbar = False):

    if LFP_measurements.size > X.size:
        LFP_measurements = LFP_measurements[:,timestep]
    LFP = np.array(LFP_measurements).reshape(X.shape)
    # normalize LFP
    LFP_norm = LFP/max_LFP

    # make contour plot
    colors, levels_norm, ticks  = get_colormap_info()
    ep_intervals = ax.contourf(X, Z, LFP_norm,# vmin=-200, vmax=200,
                               zorder=-2, colors = colors,
                               levels=levels_norm, extend = 'both') #norm = LogNorm())#,
                                          # norm = SymLogNorm(1E-30))#, vmin = -40, vmax = 40))
    ax.contour(X, Z, LFP_norm, lw = 0.4,  # 20,
               colors='k', zorder = -2,  # extend='both')
               levels=levels_norm)

    if lengthbar:
        ax.plot([-400, -400], [-200, 800], 'k', lw=2, clip_on=False)
        ax.text(-330, 400, r'$1 \mathsf{mm}$', color='k', size = 8, va='center', ha='center', rotation = 'vertical')

    ax.axis('off')
    ax.set_xlim([-500,500])
    ax.set_ylim([-250,850])
    return LFP

def plot_lfp_far(fig, ax, LFP_measurements, max_LFP, timestep, X_f, Z_f, colorax = False,
             lengthbar = False):

    if LFP_measurements.size > X_f.size:
        LFP_measurements = LFP_measurements[:,timestep]

    LFP = np.array(LFP_measurements).reshape(X_f.shape)
    # normalize LFP
    LFP_norm = LFP/max_LFP
    # make contour plot
    colors, levels_norm, ticks  = get_colormap_info()
    ep_intervals = ax.contourf(X_f, Z_f, LFP_norm,
                               zorder=-2, colors = colors,
                               levels=levels_norm, extend = 'both')
    ep_lines = ax.contour(X_f, Z_f, LFP_norm, lw = 0.4,
               colors='k', zorder = -2,
               levels=levels_norm)

    if lengthbar:
        print('lengthbar true')
        ax.plot([-14000, -14000], [-14000, -13000], 'k', lw=2, clip_on=False)
        ax.text(-11700, -13600, r'$1 \mathsf{mm}$', size = 8, va='center', ha='center')

    ax.axis('off')

    return LFP, ep_intervals, ticks

def get_colormap_info():
    num = 9
    levels = np.logspace(-4, 0, num = num)
    levels_norm = np.concatenate((-levels[::-1], levels))
    prgn_cmap = plt.cm.get_cmap('PRGn') # rainbow, spectral, RdYlBu
    colors_from_map = [prgn_cmap(i*np.int(255/(len(levels_norm) - 2))) for i in range(len(levels_norm) -1)]
    colors_from_map[num - 1] = (1.0, 1.0, 1.0, 1.0)
    ticks = [levels_norm[2*i] for i in range(int(num/2 + 1))] + [levels_norm[int(num + 2*i)] for i in range(int(num/2 + 1))]
    return colors_from_map, levels_norm, ticks

def plot_neuron(axis, syninds, xmids, zmids, zips, syn=False, lengthbar=False, clr='k', lb_clr='k'):

    # faster way to plot points:
    polycol = PolyCollection(list(zips), edgecolors = 'none', facecolors = 'k')
    axis.add_collection(polycol)
    # small length reference bar
    if lengthbar:
        axis.plot([-400, -400], [-200, 800], lb_clr, lw=2, clip_on=False)
        axis.text(-330, 400, r'$1 \mathsf{mm}$', color=lb_clr, size = 8, va='center', ha='center', rotation = 'vertical')

    axis.set_xlim([-500,500])
    axis.set_ylim([-250,850])
    axis.axis('off')

    # red dot where synapse is located
    if syn:
        for idx_num, idx in enumerate(syninds):
            axis.plot(xmids[idx], zmids[idx], 'o', ms=3,
                    markeredgecolor='k', markerfacecolor='r')

if __name__ == '__main__':
    # load data
    data = np.load('./data/data_fig2.npz')

    totnsegs = data['totnsegs']
    imem = data['imem']
    syninds = data['syninds']
    xmids = data['xmids']
    zmids = data['zmids']
    morph_zips = data['morph_zips']
    cb_LFP_close = data['cb_LFP_close']
    cb_LFP_far = data['cb_LFP_far']
    multi_dip_LFP_close = data['multi_dip_LFP_close']
    multi_dip_LFP_far = data['multi_dip_LFP_far']
    db_LFP_close = data['db_LFP_close']
    db_LFP_far = data['db_LFP_far']
    LFP_max = data['LFP_max']
    time_max = data['time_max']
    multi_dips = data['multi_dips']
    multi_dip_locs = data['multi_dip_locs']
    single_dip = data['single_dip']
    r_mid = data['r_mid']
    X = data['X']
    Z = data['Z']
    X_far = data['X_far']
    Z_far = data['Z_far']

    fig = make_fig_1(totnsegs, imem, syninds, xmids, zmids, morph_zips,
                     cb_LFP_close, cb_LFP_far,
                     multi_dip_LFP_close, multi_dip_LFP_far,
                     db_LFP_close, db_LFP_far,
                     LFP_max,
                     time_max,
                     multi_dips, multi_dip_locs,
                     single_dip, r_mid,
                     X, Z, X_far, Z_far)
    fig.savefig('./figures/Figure2.pdf', bbox_inches='tight', dpi=300)
