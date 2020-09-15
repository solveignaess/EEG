import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from matplotlib import colorbar
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.patheffects as path_effects


def plot_neuron(axis, zips = None, cell=None, syn=False, lengthbar=False):
    if cell:
        zips = []
        for x, z in cell.get_idx_polygons():
            zips.append(list(zip(x, z)))
    # faster way to plot points:
    polycol = PolyCollection(list(zips), edgecolors = 'none', facecolors = 'k')
    axis.add_collection(polycol)

    # small length reference bar
    if lengthbar:
        axis.plot([-500, -500], [-100, 900], 'k', lw=2, clip_on=False)
        axis.text(-430, 400, r'$1 \mathsf{mm}$', size = 8, va='center', ha='center', rotation = 'vertical')
    # plt.axis('tight')

if __name__ == '__main__':
    # load data from file
    # data = np.load('./data/compare_multi_single_dipole_segev_syns_from_path_passiveTrue_Segev2018_cell11.npz')
    # data = np.load('./data/data_fig2_segev_3a.npz')
    # data = np.load('./data/data_fig2_segev_3a_active_Falsestd_pms2.npz')
    data = np.load('data/data_fig2.npz')
    for item, value in data.items():
        print(item, value)

    lfp_multi_dip_list = data['lfp_multi']#[idcs_to_plot]
    lfp_single_dip_list = data['lfp_single']#[idcs_to_plot]
    p_list = data['dipoles']#[idcs_to_plot]

    sigmas = data["sigmas"]
    radii = data["radii"]
    synlocs = data['synlocs']#[idcs_to_plot]
    zips = data['zips']
    zmax = data['zmax']
    rz = data['rz']
    electrode_locs = data['electrode_locs']
    tvec = data['tvec']
    Pz_traces = data['pz_traces']#[idcs_to_plot]


    # compute values for plotting
    num_syns = len(synlocs)
    syn_loc_zs = synlocs[:,2] - rz[2]
    # dip_strength = np.linalg.norm(np.array(p_list).reshape(num_syns,3), axis=1)
    p_z = np.abs(p_list.reshape(num_syns,3)[:,2])
    # p_z = p_list.reshape(num_syns,3)[:,2]

    RE_list = np.abs((lfp_single_dip_list - lfp_multi_dip_list)/lfp_multi_dip_list) * 100 # Convert to percent
    RE_EEG = RE_list[:, -1, 0]
    k_eeg = 1e9 # convert from mV to pV
    EEG = np.array(lfp_multi_dip_list).reshape(num_syns, len(electrode_locs))[:, -1]*k_eeg

    plt.close('all')
    fig = plt.figure(figsize=[8., 7.])
    fig.subplots_adjust(wspace=0.65, hspace=0.4, left=0.09, right=0.98, bottom=0.08, top=0.97)
    # line colors
    clrs = plt.cm.viridis(np.linspace(0,0.8,num=num_syns))
    # head color
    head_colors = plt.cm.Pastel1([0,1,2,3])

    # define axes
    # ax_setup = plt.subplot2grid((3,4),(0,0), colspan=2)
    ax_setup = fig.add_axes([0.07, 0.67, 0.25, 0.29], aspect=1, frameon=False, xticks=[], yticks=[])
    ax_cdm = fig.add_axes([0.345, 0.70, 0.072, 0.11])
    ax_pot = plt.subplot2grid((3,4),(1,0), colspan=2)
    ax_pot_RE = plt.subplot2grid((3,4),(2,0), colspan=2)
    ax_RE_EEG = plt.subplot2grid((3,4),(0,2), colspan=2)
    ax_EEG = plt.subplot2grid((3,4),(1,2), colspan=2)
    ax_RE_EEG_EEG = plt.subplot2grid((3,4),(2,2), colspan=2)
    # ax_RE_ECoG_p = plt.subplot2grid((4,4),(3,2), colspan=2)

    # plot EEG amplitude as function of synapse distance from soma
    ax_EEG.scatter(syn_loc_zs, np.abs(EEG), s = 5., c = clrs)
    ax_EEG.set_xlabel(r'synapse distance from soma ($\mu$m)', fontsize=10, labelpad=0.5)
    ax_EEG.set_ylabel(r'$|$EEG$|$ (pV)', fontsize=10, labelpad=5)
    # ax_p.set_xticklabels([])

    # plot RE at EEG distance as function of synapse distance from soma
    ax_RE_EEG.scatter(syn_loc_zs, RE_EEG, s = 5., c = clrs, clip_on=False)
    ax_RE_EEG.set_xlabel(r'synapse distance from soma ($\mu$m)', fontsize=10, labelpad=0.5)
    ax_RE_EEG.set_ylabel(r'RE for EEG (%)', fontsize=10, labelpad=5)

    # plot RE at EEG distance as function of dipole strength
    ax_RE_EEG_EEG.scatter(np.abs(EEG), RE_EEG, s = 5., c = clrs, clip_on=False)
    ax_RE_EEG_EEG.set_xlabel(r'$|$EEG$|$ (pV)', fontsize=10)
    ax_RE_EEG_EEG.set_ylabel(r'RE for EEG (%)', fontsize=10, labelpad=5)
    # ax_RE_EEG_p.legend(loc=1, fontsize=6, frameon=False)


    # plot setup, potentials and RE as function of distance to electrode
    radii_tweaked = [radii[0]] + [r + 500 for r in radii[1:]]
    # plot 4s-model
    for i in range(4):
        ax_setup.add_patch(plt.Circle((0, 0), radius = radii_tweaked[-1-i], color = head_colors[-1-i], fill=True, ec = 'k', lw = .1))

    # # plot morphology with synapses
    # neuron_offset = 57000.
    plot_neuron(ax_setup, zips=zips)
    ax_setup.plot(0,0,'o', ms = 1e-4)
    # zoom in on neuron:
    zoom_ax = zoomed_inset_axes(ax_setup, 130, bbox_to_anchor=(290, 500))
    x1, x2, y1, y2 = -350, 350, radii[0]- 1250, radii[0]-150
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
    syns_to_plot = np.array([-7, 4])

    # plot p(t) mini panel
    # k_nA_to_pA = 1e3
    for i in syns_to_plot:
        ax_cdm.plot(tvec, np.abs(Pz_traces[i]), c = clrs[i], lw=1.5, clip_on=False)
        ax_RE_EEG.scatter(syn_loc_zs[i], RE_EEG[i], s = 70., c = clrs[i], clip_on=False)
        ax_EEG.scatter(syn_loc_zs[i], np.abs(EEG[i]), s = 70., c = clrs[i])
        ax_RE_EEG_EEG.scatter(np.abs(EEG[i]), RE_EEG[i], s = 70., c = clrs[i], clip_on=False)

    # ax_cdm.axvline(np.argmax(P_25), ls='--', c='gray')
    ax_cdm.set_ylabel(r'$|p_z|$ (nAµm)', fontsize = 8, labelpad=15)
    # fig.text(0.47, 0.76,'$t$', fontsize = 'xx-small')
    ax_cdm.spines['top'].set_visible(False)
    ax_cdm.spines['right'].set_visible(False)
    ax_cdm.get_xaxis().tick_bottom()
    ax_cdm.set_xticks([20, 100])
    ax_cdm.set_xticklabels([20, 100], fontsize=8)
    ax_cdm.set_xlim([0,100])
    ax_cdm.set_xlabel('t (ms)', fontsize=8)
    ax_cdm.xaxis.set_label_coords(.6, -.28)
    ax_cdm.set_yticks([10.])
    ax_cdm.set_yticklabels(['10'], fontsize=8)
    ax_cdm.set_ylim([0,10])
    ax_cdm.yaxis.set_label_coords(-.13, .43)

    electrode_locs_z = electrode_locs[:,2] - zmax
    # syns_to_plot = np.arange(0, num_syns, 5)
    k = 1e3 # from mV to uV
    k1 = 1e-3 # from mum to mm
    electrode_locs_z = electrode_locs_z*k1

    for syn in range(num_syns):
        zoom_ax.plot(synlocs[syn][0], synlocs[syn][2], 'o', color=clrs[syn], ms = 1)

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
        RE = RE_list[i].reshape(electrode_locs_z.shape)
        print('RE for synapse idx', i, 'at ECoG location:', RE[ecog_idx][0] ,'%.')
        print('RE for synapse idx', i, 'at EEG location:', RE[eeg_idx][0] ,'%.')
        ax_pot_RE.semilogx(electrode_locs_z, RE, color = clrs[i], label=str(i), linewidth=1.)
        zoom_ax.plot(synlocs[i][0], synlocs[i][2], 'o', color=clrs[i], ms=7)
    ax_pot.plot(100, 0, 'k--', label='multi-dipole', lw=0.8)
    ax_pot.plot(100, 0, 'k-', label='single-dipole', lw=0.8)
    ax_pot.legend(loc=1, fontsize=8, frameon=False)


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
    ax_pot.set_ylim([1e-7, 1e-1])
    ax_pot.set_yticks([1e-7, 1e-4, 1e-1])
    ax_pot_RE.set_ylim([0, 100])
    ax_pot.set_ylabel(r'electric potential $|\Phi|$ ($\mu$V)', fontsize=10)
    ax_pot_RE.set_ylabel(r'RE (%)', fontsize=10, labelpad=9)
    ax_pot_RE.set_xlabel(r'distance from top of neuron to electrode (mm)', fontsize=10)
    # mark ECoG and EEG locations
    plt.text(0.13, 0.65, 'ECoG', fontsize=9, transform=plt.gcf().transFigure)
    plt.text(0.47, 0.65, 'EEG', fontsize=9, transform=plt.gcf().transFigure)

    # mark 4-sphere head model layers
    txt_brain = plt.text(0.099, 0.297, 'brain', fontweight='bold', fontsize=9, transform=plt.gcf().transFigure, color='k')
    txt_csf = plt.text(0.2, 0.297, 'CSF', fontweight='bold', fontsize=9, transform=plt.gcf().transFigure, color='k')
    txt_skull = plt.text(0.35, 0.297, 'skull', fontweight='bold', fontsize=9, transform=plt.gcf().transFigure, color='k')
    txt_scalp = plt.text(0.437, 0.297, 'scalp', fontweight='bold', fontsize=9, transform=plt.gcf().transFigure, color='k')

    for ax in [ax_EEG, ax_RE_EEG]:
        ax.set_xlim([-20, 800])

    for ax in [ax_RE_EEG, ax_RE_EEG_EEG]:
        ax.set_ylim([0, 15])

    EEGmin = 0
    EEGmax = int(round(np.max(np.abs(EEG))))

    ax_RE_EEG_EEG.set_xlim([EEGmin, EEGmax+2])
    ax_RE_EEG_EEG.set_xticks([round(num) for num in range(EEGmin, EEGmax+3, 2)])
    ax_EEG.set_ylim([EEGmin, EEGmax+4])
    # ax_EEG.set_yticks([round(num) for num in range(EEGmin, EEGmax+4, 3)])

    for ax in [ax_pot, ax_pot_RE, ax_EEG, ax_RE_EEG, ax_RE_EEG_EEG]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # label axes
    xpos = [0.04, 0.04, 0.04, 0.53, 0.53, 0.53]
    ypos = [0.97, 0.68, 0.35, 0.97, 0.68, 0.35]
    letters = 'ABCDEF'
    for i in range(len(letters)):
        fig.text(xpos[i], ypos[i], letters[i],
             horizontalalignment='center',
             verticalalignment='center',
             fontweight='demibold',
             fontsize=12)

    plt.savefig('./figures/figure2.pdf', dpi=300)
