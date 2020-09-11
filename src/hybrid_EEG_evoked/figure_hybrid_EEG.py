#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from os.path import join
import numpy as np
import matplotlib.style
matplotlib.use("AGG")
matplotlib.style.use('classic')
import matplotlib.pyplot as plt
import os
import LFPy

np.random.seed(12478)
# np.random.seed(125162354)

import plotting_helpers as phlp
from hybridLFPy import CachedNetwork
import analysis_params

from params_evoked_with_EEG import multicompartment_params
from plot_methods import plot_population, plot_signal_sum

def plot_cdms(fig, params, dt, T):
    L5E_subpops = ["p5(L56)", "p5(L23)"]
    L5I_subpops = ["b5", "nb5"]
    ax_dict = {'xlim': T,
               'xticks': [],
               'yticks': [],
               'frameon': False,
               #'ylabel': "nA$\cdot\mu$m",
               'ylim': [-250, 150],
               }
    num_tsteps = 1201

    ax_cdm_Ex = fig.add_axes([0.05, 0.01, 0.08, 0.14], **ax_dict)
    ax_cdm_Ey = fig.add_axes([0.15, 0.01, 0.08, 0.14], **ax_dict)
    ax_cdm_Ez = fig.add_axes([0.26, 0.01, 0.08, 0.14], **ax_dict)

    ax_cdm_Ix = fig.add_axes([0.05, 0.15, 0.08, 0.14], **ax_dict)
    ax_cdm_Iy = fig.add_axes([0.15, 0.15, 0.08, 0.14], **ax_dict)
    ax_cdm_Iz = fig.add_axes([0.26, 0.15, 0.08, 0.14], **ax_dict)

    fig.text(0.005, 0.1, "L5E")
    fig.text(0.005, 0.23, "L5I")

    phlp.annotate_subplot(ax_cdm_Ix, ncols=4, nrows=4, letter='D', linear_offset=0.065)

    ax_list = [ax_cdm_Ex, ax_cdm_Ey, ax_cdm_Ez, ax_cdm_Ix, ax_cdm_Iy, ax_cdm_Iz]
    ax_name = ["P$_x$", "P$_y$", "P$_z$", "P$_x$", "P$_y$", "P$_z$"]
    for i, ax in enumerate(ax_list):
        phlp.remove_axis_junk(ax)
        ax.axvline(900, c='gray', ls='--')
        ax.set_rasterized(True)
    # for ax in [ax_cdm_Ex, ax_cdm_Ey, ax_cdm_Ez]:
        ax.text(870, 150, ax_name[i])
        ax.plot([920, 920], [60, 160], c='k', lw=1)
        if i == 2:
            ax.text(925, 85, "100 nA$\mu$m")
        # ax.set_ylabel("nA$\cdot\mu$m", labelpad=-9)

    t = np.arange(num_tsteps) * dt
    summed_cdm_E = np.zeros((len(t), 3))
    summed_cdm_I = np.zeros((len(t), 3))

    tot_num_E = 0

    linedict = dict(lw=0.5, c="0.7", clip_on=True, zorder=-1)

    for subpop in L5E_subpops:
        cdm_folder = join(params.savefolder, "cdm", "{}".format(subpop))
        files = os.listdir(cdm_folder)

        for idx, f in enumerate(files):
            cdm = np.load(join(cdm_folder, f))[:, :]
            summed_cdm_E += cdm
            tot_num_E += 1
            # if idx < 100:
            ax_cdm_Ex.plot(t, cdm[:, 0], **linedict)
            ax_cdm_Ey.plot(t, cdm[:, 1], **linedict)
            ax_cdm_Ez.plot(t, cdm[:, 2], **linedict)
    summed_cdm_E /= tot_num_E

    tot_num_I = 0
    for subpop in L5I_subpops:
        cdm_folder = join(params.savefolder, "cdm", "{}".format(subpop))
        files = os.listdir(cdm_folder)

        for idx, f in enumerate(files):
            cdm = np.load(join(cdm_folder, f))[:, :]
            cdm -= np.average(cdm, axis=0)
            summed_cdm_I += cdm
            tot_num_I += 1
            # if idx < 100:
            ax_cdm_Ix.plot(t, cdm[:, 0], **linedict)
            ax_cdm_Iy.plot(t, cdm[:, 1], **linedict)
            ax_cdm_Iz.plot(t, cdm[:, 2], **linedict)

    summed_cdm_I /= tot_num_I

    ax_cdm_Ex.plot(t, summed_cdm_E[:, 0], lw=1, c="k")
    ax_cdm_Ey.plot(t, summed_cdm_E[:, 1], lw=1, c="k")
    ax_cdm_Ez.plot(t, summed_cdm_E[:, 2], lw=1, c="k")

    ax_cdm_Ix.plot(t, summed_cdm_I[:, 0], lw=1, c="k")
    ax_cdm_Iy.plot(t, summed_cdm_I[:, 1], lw=1, c="k")
    ax_cdm_Iz.plot(t, summed_cdm_I[:, 2], lw=1, c="k")


def fig_intro(params, fraction=0.05, rasterized=False):
    '''set up plot for introduction'''
    plt.close("all")

    #load spike as database
    networkSim = CachedNetwork(**params.networkSimParams)
    # num_pops = 8

    fig = plt.figure(figsize=[4.5, 3.5])

    fig.subplots_adjust(left=0.03, right=0.98, wspace=0.5, hspace=0.)
    ax_spikes = fig.add_axes([0.09, 0.4, 0.2, 0.55])
    ax_morph = fig.add_axes([0.37, 0.3, 0.3, 0.75], frameon=False, aspect=1,
                            xticks=[], yticks=[])
    ax_lfp = fig.add_axes([0.73, 0.4, 0.23, 0.55], frameon=True)
    ax_4s = fig.add_axes([0.42, 0.05, 0.25, 0.2], frameon=False, aspect=1,
                         title='head model', xticks=[], yticks=[])
    ax_top_EEG = fig.add_axes([0.65, 0.02, 0.33, 0.32], frameon=False, xticks=[], yticks=[],
                              ylim=[-0.7, .3])
    dt = 1
    t_idx = 875
    T = [t_idx, t_idx + 75]

    fig.text(0.55, 0.97, "multicompartment neurons", fontsize=6, ha="center")
    ax_spikes.set_title("spiking activity", fontsize=6)
    ax_lfp.set_title("LFP", fontsize=6)

    #network raster
    ax_spikes.xaxis.set_major_locator(plt.MaxNLocator(4))
    phlp.remove_axis_junk(ax_spikes)
    phlp.annotate_subplot(ax_spikes, ncols=4, nrows=1, letter='A', linear_offset=0.045)
    x, y = networkSim.get_xy(T, fraction=fraction)

    networkSim.plot_raster(ax_spikes, T, x, y, markersize=0.2, marker='_',
                           alpha=1., legend=False, pop_names=True, rasterized=rasterized)

    #population
    plot_population(ax_morph, params, isometricangle=np.pi/24, plot_somas=False,
                    plot_morphos=True, num_unitsE=1, num_unitsI=1,
                    clip_dendrites=True, main_pops=True, title='',
                    rasterized=rasterized)
    # ax_morph.set_title('multicompartment neurons', va='top')
    phlp.annotate_subplot(ax_morph, ncols=5, nrows=1, letter='B', linear_offset=0.005)

    phlp.remove_axis_junk(ax_lfp)
    #ax_lfp.set_title('LFP', va='bottom')
    ax_lfp.xaxis.set_major_locator(plt.MaxNLocator(4))
    phlp.annotate_subplot(ax_lfp, ncols=4, nrows=2, letter='C', linear_offset=0.025)
    #print(ax_morph.axis())
    plot_signal_sum(ax_lfp, params, fname=join(params.savefolder, 'LFPsum.h5'),
                unit='mV', vlimround=0.8,
                T=T, ylim=[-1600, 100],
                rasterized=False)

    plot_cdms(fig, params, dt, T)

    plot_foursphere_to_ax(ax_4s)
    phlp.annotate_subplot(ax_4s, ncols=3, nrows=7, letter='E', linear_offset=0.05)

    # Plot EEG at top of head
    # ax_top_EEG.xaxis.set_major_locator(plt.MaxNLocator(4))
    phlp.annotate_subplot(ax_top_EEG, ncols=1, nrows=1, letter='F', linear_offset=-0.08)

    # ax_top_EEG.set_ylabel("$\mu$V", labelpad=-3)
    summed_top_EEG = np.load(join(params.savefolder, "summed_EEG.npy"))

    simple_EEG_single_pop = np.load(join(params.savefolder, "simple_EEG_single_pop.npy"))
    simple_EEG_pops_with_pos = np.load(join(params.savefolder, "simple_EEG_pops_with_pos.npy"))

    tvec = np.arange(len(summed_top_EEG)) * dt

    # sub_pops = ["L5I", "L4I", "L6I", "L23I", "L5E", "L4E", "L6E", "L23E"]
    pops = np.unique(next(zip(*params.mapping_Yy)))
    colors = phlp.get_colors(np.unique(pops).size)
    for p_idx, pop in enumerate(pops):
        pop_eeg = np.load(join(params.savefolder, "EEG_{}.npy".format(pop)))
        pop_eeg -= np.average(pop_eeg)
        # pop_sum.append(pop_eeg)
        ax_top_EEG.plot(pop_eeg, c=colors[p_idx], lw=1)

    ax_top_EEG.plot([878, 878], [-0.1, -0.3], c='k', lw=1)
    ax_top_EEG.plot([878, 888], [-0.3, -0.3], c='k', lw=1)
    ax_top_EEG.text(879, -0.2, "0.2 $\mu$V", va="center")
    ax_top_EEG.text(885, -0.32, "10 ms", va="top", ha="center")

    y0 = summed_top_EEG - np.average(summed_top_EEG)
    y1 = simple_EEG_single_pop - np.average(simple_EEG_single_pop)
    y2 = simple_EEG_pops_with_pos - np.average(simple_EEG_pops_with_pos)

    l3, = ax_top_EEG.plot(tvec, y0-y2, lw=1.5, c='orange', ls='-')
    l1, = ax_top_EEG.plot(tvec, y0, lw=1.5, c='k')
    l2, = ax_top_EEG.plot(tvec, y2, lw=1.5, c='r', ls='--')

    t0_plot_idx = np.argmin(np.abs(tvec - 875))
    t1_plot_idx = np.argmin(np.abs(tvec - 950))
    max_sig_idx = np.argmax(np.abs(y0[t0_plot_idx:])) + t0_plot_idx

    EEG_error_at_max_1 = np.abs(y0[max_sig_idx] - y1[max_sig_idx]) / np.abs(y0[max_sig_idx])
    EEG_error_at_max_2 = np.abs(y0[max_sig_idx] - y2[max_sig_idx]) / np.abs(y0[max_sig_idx])

    max_EEG_error_1 = np.max(np.abs(y0[t0_plot_idx:t1_plot_idx] - y1[t0_plot_idx:t1_plot_idx]) / np.max(np.abs(y0[t0_plot_idx:t1_plot_idx])))
    max_EEG_error_2 = np.max(np.abs(y0[t0_plot_idx:t1_plot_idx] - y2[t0_plot_idx:t1_plot_idx]) / np.max(np.abs(y0[t0_plot_idx:t1_plot_idx])))

    print("Error with single pop at sig max (t={:1.3f} ms): {:1.4f}. Max relative error: {:1.4f}".format(tvec[max_sig_idx], EEG_error_at_max_1, max_EEG_error_1))
    print("Error with multipop at sig max (t={:1.3f} ms): {:1.4f}. Max relative error: {:1.4f}".format(tvec[max_sig_idx], EEG_error_at_max_2, max_EEG_error_2))
    ax_top_EEG.legend([l1, l2, l3], ["full sum", "pop. dipole", "difference"], frameon=False, loc=(0.5, 0.1))
    # phlp.remove_axis_junk(ax_top_EEG)
    ax_top_EEG.axvline(900, c='gray', ls='--')
    ax_lfp.axvline(900, c='gray', ls='--')

    ax_top_EEG.set_xlim(T)

    fig.savefig(join('hybrid_with_EEG.png'), dpi=300)
    fig.savefig(join('hybrid_with_EEG.pdf'), dpi=300)


def plot_foursphere_to_ax(ax):

    # four_sphere properties
    radii = [79000., 80000., 85000., 90000.]
    radii_name = ["Cortex", "CSF", "Skull", "Scalp"]

    xlim = [-7000, 7000]
    ylim = [radii[0]-6000, radii[-1] + 100]

    max_angle = np.abs(np.rad2deg(np.arcsin(xlim[0] / ylim[0])))

    pop_bottom = radii[0] - 1500
    pop_top = radii[0]

    isometricangle = np.pi/24
    r = 564

    theta0 = np.linspace(0, np.pi, 20)
    theta1 = np.linspace(np.pi, 2*np.pi, 20)

    outline_params = dict(color='red', lw=0.5, zorder=50)
    ax.plot(r*np.cos(theta0), r*np.sin(theta0)*np.sin(isometricangle)+pop_bottom,
            **outline_params)
    ax.plot(r*np.cos(theta1), r*np.sin(theta1)*np.sin(isometricangle)+pop_bottom,
            **outline_params)
    ax.plot(r*np.cos(theta0), r*np.sin(theta0)*np.sin(isometricangle)+pop_top,
            **outline_params)
    ax.plot(r*np.cos(theta1), r*np.sin(theta1)*np.sin(isometricangle)+pop_top,
            **outline_params)

    ax.plot([-r, -r], [pop_bottom, pop_top], **outline_params)
    ax.plot([r, r], [pop_bottom, pop_top], **outline_params)

    angle = np.linspace(-max_angle, max_angle, 100)
    for b_idx in range(len(radii)):
        x_ = radii[b_idx] * np.sin(np.deg2rad(angle))
        z_ = radii[b_idx] * np.cos(np.deg2rad(angle))
        l_curved, = ax.plot(x_, z_, c="k", lw=1)
        z_shift = -500 if b_idx == 0 else 0
        ax.text(x_[0], z_[0] + z_shift, radii_name[b_idx], fontsize=6,
                va="top", ha="right", color="k", rotation=5)

    ax.plot([2500, 7500], [radii[0] - 3000, radii[0] - 3000], c='gray', lw=1)
    ax.text(2500, radii[0] - 2500, "5 mm", color="gray")

    ax.plot([-300, 300], [radii[-1], radii[-1]], 'b', lw=2)
    # ax.text(0, radii[-1] + 500, "EEG electrode", ha="center")


if __name__ == '__main__':
    plt.close('all')

    params = multicompartment_params()
    ana_params = analysis_params.params()

    savefolder = 'evoked_cdm'

    params.savefolder = os.path.join(os.path.split(params.savefolder)[0],
                                     savefolder)
    params.figures_path = os.path.join(params.savefolder, 'figures')
    params.spike_output_path = os.path.join(params.savefolder,
                                            'processed_nest_output')
    params.networkSimParams['spike_output_path'] = params.spike_output_path

    fig_intro(params, fraction=1.)
