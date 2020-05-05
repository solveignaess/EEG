import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colorbar
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset



def plot_neuron(axis, zips = None, cell=None, syn=False, lengthbar=False):
    if cell:
        zips = []
        for x, z in cell.get_idx_polygons():
            zips.append(list(zip(x, z)))
    # faster way to plot points:
    polycol = PolyCollection(list(zips), edgecolors = 'none', facecolors = 'k')
    axis.add_collection(polycol)


    plt.axis('tight')

    # small length reference bar
    if lengthbar:
        axis.plot([-500, -500], [-100, 900], 'k', lw=2, clip_on=False)
        axis.text(-430, 400, r'$1 \mathsf{mm}$', size = 8, va='center', ha='center', rotation = 'vertical')
    plt.axis('tight')

if __name__ == '__main__':
    # load data from file
    # data = np.load('./data/compare_multi_single_dipole_segev_syns_from_path_passiveTrue_Segev2018_cell11.npz')
    # data = np.load('./data/data_fig2_segev_3a.npz')
    data = np.load('./data/data_fig2_segev_3aactive_Falsestd_pms2.npz')
    # data = np.load('./data/data_fig2_segev_8.npz')
    # data = np.load('./data/compare_multi_single_dipole_segev_syns_from_path_passiveTrue_new_diploc.npz')
    # data = np.load('./data/compare_multi_single_dipole_segev_syns_from_path_passiveTrue.npz')
    # data = np.load('./data/compare_multi_single_dipole_Hay_syns_from_path_passiveTrue.npz')
    # idcs_to_plot = [i for i in range(0,41)]  #[0, 1, 2] + [i for i in range(4,41)]
    lfp_multi_dip_list = data['lfp_multi']#[idcs_to_plot]
    lfp_single_dip_list = data['lfp_single']#[idcs_to_plot]
    RE_EEG = data['re_eeg']#[idcs_to_plot]
    RE_ECoG = data['re_ecog']#[idcs_to_plot]
    p_list = data['dipoles']#[idcs_to_plot]
    p_loc_list = data['dip_locs']#[idcs_to_plot]
    sigmas = data['sigmas']
    radii = data['radii']
    synlocs = data['synlocs']#[idcs_to_plot]
    # RE_EEG_30syns = data['RE_EEG_30syns']
    zips = data['zips']
    zmax = data['zmax']
    rz = data['rz']
    electrode_locs = data['electrode_locs']
    tvec = data['tvec']
    t_max_list = data['t_max_list']
    Pz_traces = data['Pz_traces']#[idcs_to_plot]
    # data_pz = np.load('./data/pz_segev_2.npz')
    # Pz_traces = data_pz['Pz_traces']#[idcs_to_plot,:1601]

    # compute values for plotting
    num_syns = len(synlocs)
    syn_loc_zs = synlocs[:,2] - rz[2]
    # dip_strength = np.linalg.norm(np.array(p_list).reshape(num_syns,3), axis=1)
    p_z = np.abs(p_list.reshape(num_syns,3)[:,2])
    # p_z = p_list.reshape(num_syns,3)[:,2]

    RE_list = np.abs((lfp_single_dip_list - lfp_multi_dip_list)/lfp_multi_dip_list)

    k_eeg = 1e9 # convert from mV to pV
    EEG = np.array(lfp_multi_dip_list).reshape(num_syns, len(electrode_locs))[:, -1]*k_eeg

    plt.close('all')
    fig = plt.figure()
    # line colors
    clrs = plt.cm.viridis(np.linspace(0,0.8,num=num_syns))
    # head color
    head_colors = plt.cm.Pastel1([0,1,2,3])

    # define axes
    # ax_setup = plt.subplot2grid((3,4),(0,0), colspan=2)
    ax_setup = fig.add_axes([0.07, 0.67, 0.25, 0.27])
    ax_cdm = fig.add_axes([0.3455, 0.675, 0.063, 0.09])
    ax_pot = plt.subplot2grid((3,4),(1,0), colspan=2)
    ax_pot_RE = plt.subplot2grid((3,4),(2,0), colspan=2)
    ax_RE_EEG = plt.subplot2grid((3,4),(0,2), colspan=2)
    ax_EEG = plt.subplot2grid((3,4),(1,2), colspan=2)
    ax_RE_EEG_EEG = plt.subplot2grid((3,4),(2,2), colspan=2)
    # ax_RE_ECoG_p = plt.subplot2grid((4,4),(3,2), colspan=2)

    # # plot dipole strength as function of synapse distance from soma
    # ax_p.scatter(syn_loc_zs, p_z, s = 5., c = clrs)
    # ax_p.set_xlabel(r'synapse distance from soma ($\mu$m)', fontsize=8, labelpad=0.5)
    # ax_p.set_ylabel(r'dipole moment $|\mathbf{p}_z|$ (nA$\mu$m)', fontsize=8, labelpad=5)
    # # ax_p.set_xticklabels([])

    # plot EEG amplitude as function of synapse distance from soma
    ax_EEG.scatter(syn_loc_zs, np.abs(EEG), s = 5., c = clrs)
    ax_EEG.set_xlabel(r'synapse distance from soma ($\mu$m)', fontsize=8, labelpad=0.5)
    ax_EEG.set_ylabel(r'$|$EEG$|$ (pV)', fontsize=8, labelpad=5)
    # ax_p.set_xticklabels([])

    # plot RE at EEG distance as function of synapse distance from soma
    ax_RE_EEG.scatter(syn_loc_zs, RE_EEG, s = 5., c = clrs, clip_on=False)
    ax_RE_EEG.set_xlabel(r'synapse distance from soma ($\mu$m)', fontsize=8, labelpad=0.5)
    ax_RE_EEG.set_ylabel(r'RE for EEG (%)', fontsize=8, labelpad=5)

    # plot RE at EEG distance as function of dipole strength
    # ax_RE_EEG_p.plot(np.arange(16), np.ones(16)*RE_EEG_30syns, 'k--', label="all synapses active simultaneously")
    # ax_RE_EEG_p.scatter(dip_strength, RE_EEG, s = 5., c = clrs)
    ax_RE_EEG_EEG.scatter(np.abs(EEG), RE_EEG, s = 5., c = clrs, clip_on=False)
    ax_RE_EEG_EEG.set_xlabel(r'$|$EEG$|$ (pV)', fontsize=8)
    ax_RE_EEG_EEG.set_ylabel(r'RE for EEG (%)', fontsize=8, labelpad=5)
    # ax_RE_EEG_p.legend(loc=1, fontsize=6, frameon=False)


    # # plot RE at ECoG distance as function of dipole strength
    # ax_RE_ECoG_p.scatter(dip_strength, RE_ECoG, s = 5., c = clrs)
    # ax_RE_ECoG_p.set_xlabel(r'dipole strength $|p|$ (nA$\mu$m)', fontsize=7)
    # ax_RE_ECoG_p.set_ylabel(r'RE at ECoG distance (%)', fontsize=7)

    # plot setup, potentials and RE as function of distance to electrode
    radii_tweaked = [radii[0]] + [r + 500 for r in radii[1:]]
    # plot 4s-model
    for i in range(4):
        ax_setup.add_patch(plt.Circle((0, 0), radius = radii_tweaked[-1-i], color = head_colors[-1-i], fill=True, ec = 'k', lw = .1))

    # # plot morphology with synapses
    neuron_offset = 57000.
    plot_neuron(ax_setup, zips=zips)
    ax_setup.plot(0,0,'o', ms = 1e-4)
    # zoom in on neuron:
    zoom_ax = zoomed_inset_axes(ax_setup, 110, bbox_to_anchor=(2000, 3500)) # zoom = 6
    x1, x2, y1, y2 = -1000, 1000, 76000, 79200
    zoom_ax.set_facecolor(head_colors[0])
    zoom_ax.set_xlim(x1, x2)
    zoom_ax.set_ylim(y1, y2)
    zoom_ax.xaxis.set_visible('True')
    plot_neuron(zoom_ax, zips=zips, syn=True)
    mark_inset(ax_setup, zoom_ax, loc1=2, loc2=3, fc="None", ec=".5", lw=.4)
    [i.set_linewidth(.6) for i in zoom_ax.spines.values()]

    zoom_ax.xaxis.set_ticks_position('none')
    zoom_ax.xaxis.set_ticklabels([])
    zoom_ax.yaxis.set_ticks_position('none')
    zoom_ax.yaxis.set_ticklabels([])

    # syns_to_plot = np.array([-9, 2])
    syns_to_plot = np.array([-7, 6])

    # plot p(t) mini panel
    for i in syns_to_plot:
        ax_cdm.plot(tvec, np.abs(Pz_traces[i]), c = clrs[i], lw=1)
    # ax_cdm.axvline(np.argmax(P_25), ls='--', c='gray')
    ax_cdm.set_ylabel(r'$|\mathbf{p}_z|$', fontsize = 'x-small')
    # fig.text(0.47, 0.76,'$t$', fontsize = 'xx-small')
    ax_cdm.spines['top'].set_visible(False)
    ax_cdm.spines['right'].set_visible(False)
    ax_cdm.get_xaxis().tick_bottom()
    ax_cdm.set_xticks([])
    # ax_cdm.set_xticklabels([r'$t_{p max}$'], fontsize = 'xx-small')
    ax_cdm.set_xlabel('t', fontsize='x-small')
    ax_cdm.set_yticks([])
    ax_cdm.set_yticklabels([])
    # ax_cdm.annotate('', xy = (0,1), xycoords=('data', 'axes fraction'),
    #                 textcoords='offset points',
    #                 arrowprops=dict(arrowstyle='<-', facecolor='black'))
    # ax_cdm.annotate('', xy = (20,0), xycoords=('data', 'axes fraction'),
    #                 textcoords='offset points',
    #                 arrowprops=dict(arrowstyle='<-', facecolor='black'))


    plt.axis('tight')
    ax_setup.axis('off')
    ax_setup.set_aspect('equal')

    electrode_locs_z = electrode_locs[:,2] - zmax
    # syns_to_plot = np.arange(0, num_syns, 5)
    k = 1e3 # from mV to uV
    k1 = 1e-3 # from mum to mm
    electrode_locs_z = electrode_locs_z*k1

    for syn in range(num_syns):
        zoom_ax.plot(synlocs[syn][0], synlocs[syn][2], 'o', color=clrs[syn], ms = .5)

    k_100 = 100
    layer_dist_from_neuron = [(r - zmax)*k1 for r in radii]
    ecog_idx = np.where(np.abs(electrode_locs_z - layer_dist_from_neuron[0]) < 1e-3)
    eeg_idx = np.where(np.abs(electrode_locs_z - layer_dist_from_neuron[3]) < 1e-3)
    for i in syns_to_plot:
        # plot lfps
        lfp_single_dip = lfp_single_dip_list[i].reshape(electrode_locs_z.shape)*k
        lfp_multi_dip = lfp_multi_dip_list[i].reshape(electrode_locs_z.shape)*k
        ax_pot.loglog(electrode_locs_z, np.abs(lfp_single_dip), color=clrs[i], linewidth=1.)
        ax_pot.loglog(electrode_locs_z, np.abs(lfp_multi_dip), '--', color=clrs[i], linewidth=1.)

        # plot relative errors
        RE = RE_list[i].reshape(electrode_locs_z.shape)*k_100
        print('RE for synapse idx', i, 'at ECoG location:', RE[ecog_idx][0] ,'%.')
        print('RE for synapse idx', i, 'at EEG location:', RE[eeg_idx][0] ,'%.')
        ax_pot_RE.semilogx(electrode_locs_z, RE, color = clrs[i], label=str(i), linewidth=1.)
        zoom_ax.plot(synlocs[i][0], synlocs[i][2], 'o', color=clrs[i], ms = 3)
    ax_pot.plot(100, 0, 'k--', label='multi-dipole', lw=0.8)
    ax_pot.plot(100, 0, 'k-', label='single-dipole', lw=0.8)
    ax_pot.legend(loc=1, fontsize=6, frameon=False)


    # fix axes
    layer_dist_from_neuron = [(r - zmax)*k1 for r in radii]
    for ax in [ax_pot, ax_pot_RE]:
        # ax.set_xticks(list(ax.get_xticks()) + [500, 2000, 5000])
        ax.set_xlim([np.min(electrode_locs_z), np.max(electrode_locs_z)])
        ax.axvspan(0, layer_dist_from_neuron[0], facecolor=head_colors[0])
        ax.axvspan(layer_dist_from_neuron[0], layer_dist_from_neuron[1], facecolor=head_colors[1])
        ax.axvspan(layer_dist_from_neuron[1], layer_dist_from_neuron[2], facecolor=head_colors[2])
        ax.axvspan(layer_dist_from_neuron[2], layer_dist_from_neuron[3], facecolor=head_colors[3])
        ax.axvline(layer_dist_from_neuron[0], linestyle='--', linewidth=0.6, color='gray')
        ax.axvline(layer_dist_from_neuron[3]-0.13, linestyle='--', linewidth=0.6, color='gray')
    # ax_pot.set_xticklabels([])
    ax_pot.set_ylim([1e-6, 1e-1])
    ax_pot.set_yticks([1e-5, 1e-3, 1e-1])
    ax_pot_RE.set_ylim([0, 100])
    ax_pot.set_ylabel(r'electric potential $|\Phi|$ ($\mu$V)', fontsize=8)
    ax_pot_RE.set_ylabel(r'RE (%)', fontsize=8, labelpad=9)
    ax_pot_RE.set_xlabel(r'distance from top of neuron to electrode (mm)', fontsize=8)
    # mark ECoG and EEG locations
    plt.text(0.147, 0.616, 'ECoG', fontsize=8, transform=plt.gcf().transFigure)
    plt.text(0.467, 0.616, 'EEG', fontsize=8, transform=plt.gcf().transFigure)

    # mark 4-sphere head model layers
    plt.text(0.117, 0.32, 'brain', fontweight='bold', fontsize=8, transform=plt.gcf().transFigure, color=head_colors[0])
    plt.text(0.218, 0.32, 'CSF', fontweight='bold', fontsize=8, transform=plt.gcf().transFigure, color=head_colors[1])
    plt.text(0.354, 0.32, 'skull', fontweight='bold', fontsize=8, transform=plt.gcf().transFigure, color=head_colors[2])
    plt.text(0.439, 0.32, 'scalp', fontweight='bold', fontsize=8, transform=plt.gcf().transFigure, color=head_colors[3])

    for ax in [ax_EEG, ax_RE_EEG]:
        ax.set_xlim([-20, 800])

    for ax in [ax_RE_EEG, ax_RE_EEG_EEG]:
    #     ax.set_ylim([0, 20])
        ax.set_ylim([0, 15])

    # pmin = 0
    # pmax = int(round(np.max(p_z) + 1))
    #
    # ax_RE_EEG_p.set_xlim([pmin, pmax])
    # ax_RE_EEG_p.set_xticks([round(num) for num in range(pmin, pmax+1, 3)])
    # ax_p.set_ylim([pmin, pmax])
    # ax_p.set_yticks([round(num) for num in range(pmin, pmax+1, 3)])

    EEGmin = 0
    EEGmax = int(round(np.max(np.abs(EEG))))

    ax_RE_EEG_EEG.set_xlim([EEGmin, EEGmax+2])
    ax_RE_EEG_EEG.set_xticks([round(num) for num in range(EEGmin, EEGmax+3, 2)])
    ax_EEG.set_ylim([EEGmin, EEGmax+4])
    # ax_EEG.set_yticks([round(num) for num in range(EEGmin, EEGmax+4, 3)])


    for ax in [ax_pot, ax_pot_RE, ax_EEG, ax_RE_EEG, ax_RE_EEG_EEG]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    fig.tight_layout(pad=0.5, h_pad=-.9, w_pad=1.1)
    fig.set_size_inches(8., 6.)
    fig.subplots_adjust(bottom=.08, top=.91)
    # plotting_convention.mark_subplots(fig.axes[:-1], xpos=-0.25)

    # label axes
    xpos = [0.04, 0.04, 0.04, 0.541, 0.542, 0.542]
    ypos = [0.945, 0.64, 0.325, 0.945, 0.64, 0.325]
    letters = 'ABCDEF'
    for i in range(len(letters)):
        fig.text(xpos[i], ypos[i], letters[i],
             horizontalalignment='center',
             verticalalignment='center',
             fontweight='demibold',
             fontsize=8)

    # plt.savefig('./figures/figure2_passiveTrue_Hay.png', dpi=600)
    # plt.savefig('./figures/figure2_passiveTrue_segev_new_diploc.png', dpi=600)
    plt.savefig('./figures/figure2_eeg.png', dpi=600)
