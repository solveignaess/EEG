import numpy as np
import matplotlib.pyplot as plt
import plotting_convention as plotting_convention
import LFPy
import matplotlib
from matplotlib import colorbar
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from fig_dipole_field import make_data, make_fig_1
import neuron

def return_path_to_tip_idcs(cell, pos_x, pos_y, pos_z, section='allsec'):

    tuft_tip_idx = cell.get_closest_idx(x=pos_x, y=pos_y, z=pos_z, section=section)
    tuft_tip_name = cell.get_idx_name(tuft_tip_idx)[1]
    # print(tuft_tip_idx, tuft_tip_name)

    tuft_tip_sec = None
    for sec in neuron.h.allsec():
        if sec.name() == tuft_tip_name:
            tuft_tip_sec = sec
        # print(sec.name(), sec.parentseg())
        # print(sec.parentseg())

    curr_sec = tuft_tip_sec
    path_to_tip_sections = [tuft_tip_sec.name()]
    for _ in range(100):
        print(curr_sec.name())
        curr_sec = curr_sec.parentseg().sec
        path_to_tip_sections.append(curr_sec.name())
        if "soma" in curr_sec.name():
            break
    path_to_tip_idcs = []
    for secname in path_to_tip_sections:
        for idx in cell.get_idx(secname):
            path_to_tip_idcs.append(idx)
    path_to_tip_idcs.sort()
    return path_to_tip_idcs

def set_parameters():
    """set cell, synapse and electrode parameters"""
    cell_parameters = {'morphology': './cell_models/segev/CNG_version/2013_03_06_cell03_789_H41_03.CNG.swc', #'./cell_models/hay/L5bPCmodelsEH/morphologies/cell1.asc',# only mandatory parameter #
                   'tstart': 0., # simulation start time
                   'tstop': 100 # simulation stop time [ms]
                   }
    # default time resolution for NEURON and Python is 0.1!

    synapse_parameters = {'e': 0., # reversal potential
                      'syntype': 'ExpSyn', # exponential synapse
                      'tau': 5., # synapse time constant
                      'weight': 0.001, # 0.001, # synapse weight
                      'record_current': True # record synapse current
                      }
    return cell_parameters, synapse_parameters

def simulate(synidx, celltype='l23'):
    """set synapse location. simulate cell, synapse and electrodes for input synapse location"""

    # create cell with parameters in dictionary
    cell = LFPy.Cell(**cell_parameters)
    if celltype == 'l23':
        cell.set_rotation(x=np.pi/2)

    for idx in synidx:
        synapse_parameters['idx'] = idx #cell.get_closest_idx(x=pos[0], y=pos[1], z=pos[2])
        # create synapse with parameters in dictionary
        synapse = LFPy.Synapse(cell, **synapse_parameters)
        synapse.set_spike_times(np.array([20.]))
        #  simulation goes from t: 0-100 in ms. spike_time = 20ms
        # timeres = 0.1 --> 801 measurements!

    cell.simulate(rec_imem = True,
                  rec_vmem = True,
                  rec_current_dipole_moment=True)

    #create grid electrodes
    electrode_array = LFPy.RecExtElectrode(cell, **electrodeParams)
    electrode_array.calc_lfp()

    return cell, synapse, electrode_array

def get_dipole_loc(rz, syn_loc):
    mid_pos = np.array([syn_loc[0], syn_loc[1], syn_loc[2]])/2
    dip_loc = rz + mid_pos
    return dip_loc

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
    # make 4S-parameters
    sigmas = [0.3, 1.5, 0.015, 0.3]  #
    radii = [79000., 80000., 85000., 90000.]
    # set soma position
    rz = np.array([0., 0., 77500.]) #NB!!!!!!

    # found syn_idcs along same path for segev cell this way
    # syn_idcs = return_path_to_tip_idcs(cell, 0., 0., 775, section='allsec')
    syn_idcs = [0, 192, 206, 222, 224, 226, 238, 252, 270, 284, 294, 298, 304,
                308, 309, 310, 318, 319, 320, 321, 322, 323, 324, 325, 326, 330,
                331, 332, 333, 334, 338, 339, 340]
    num_syns = len(syn_idcs)

    # make electrode array params
    num_electrodes = 40
    electrode_loc_zs = list(np.linspace(78700., radii[-1], num_electrodes))
    electrode_loc_zs.insert(1, radii[0])
    electrode_loc_zs.sort()
    num_electrodes += 1
    electrode_locs = np.zeros((num_electrodes, 3))

    electrode_locs[:,2] = np.array(electrode_loc_zs)

    electrodeParams = {'sigma': 0.3,
                        'x': electrode_locs[:,0],
                        'y': electrode_locs[:,1],
                        'z': electrode_locs[:,2],
                        }

    # set cell and synapse parameters
    celltype = 'l5'
    cell_parameters, synapse_parameters = set_parameters()
    # create four-sphere class instance
    fs = LFPy.FourSphereVolumeConductor(radii, sigmas, electrode_locs)
    # lists for storing data:
    p_list = []
    p_loc_list = []
    lfp_single_dip_list = []
    lfp_multi_dip_list = []
    RE_list = []
    synlocs = []
    # get data from num_syns simulations
    for i in range(num_syns):
        print('syn number:', i)
        syn_idx = syn_idcs[i]
        # syn_loc = syn_locs[i]
        cell, synapse, electrode_array = simulate([syn_idx])
        print('cell simulated')
        syn_loc = (cell.xmid[syn_idx], cell.ymid[syn_idx], cell.zmid[syn_idx])
        cell.set_pos(x=rz[0], y=rz[1], z=rz[2])

        # compute timepoint with biggest dipole
        dipoles = cell.current_dipole_moment
        timemax = [np.argmax(np.linalg.norm(np.abs(dipoles),axis=1))]
        p = dipoles[timemax]
        # compute LFP with single dipole
        dip_loc = get_dipole_loc(rz, syn_loc)
        lfp_single_dip = fs.calc_potential(p, dip_loc)
        print('pot from single dip computed')
        # compute LFP with multi-dipole
        lfp_multi_dip = fs.calc_potential_from_multi_dipoles(cell, timemax)
        print('pot from multi dip computed')
        # compute relative errors
        RE = np.abs((lfp_single_dip - lfp_multi_dip)/lfp_multi_dip)

        synlocs.append((syn_loc[0], syn_loc[1], syn_loc[2]+rz[2]))
        p_list.append(p)
        p_loc_list.append(dip_loc)
        lfp_single_dip_list.append(lfp_single_dip)
        lfp_multi_dip_list.append(lfp_multi_dip)
        RE_list.append(RE)
        ## uncomment if you want to make fig1 for each synapse location
        # cell1, cb_LFP_close, cb_LFP_far, multi_dip_LFP_close, multi_dip_LFP_far, db_LFP_close, db_LFP_far, LFP_max_close, LFP_max_far, time_max, multi_dips, multi_dip_locs, single_dip, r_mid, X, Z, X_f, Z_f = make_data('./cell_models/segev/CNG_version/2013_03_06_cell03_789_H41_03.CNG.swc', syn_loc)
        # fig = make_fig_1(cell1,
        #                  cb_LFP_close, cb_LFP_far,
        #                  multi_dip_LFP_close, multi_dip_LFP_far,
        #                  db_LFP_close, db_LFP_far,
        #                  LFP_max_close, LFP_max_far,
        #                  time_max,
        #                  multi_dips, multi_dip_locs,
        #                  single_dip, r_mid,
        #                  X, Z, X_f, Z_f)
        # fig1_title = './figures/test_figs/fig_dipole_field' + str(i) + '.png'
        # fig.savefig(fig1_title, bbox_inches='tight', dpi=300, transparent=True)
    k_100 = 100 # convert to percentage
    eeg_ind = -1
    ecog_ind = 2
    RE_EEG = np.array(RE_list).reshape(num_syns, num_electrodes)[:,eeg_ind]*k_100
    RE_ECoG = np.array(RE_list).reshape(num_syns, num_electrodes)[:,ecog_ind]*k_100
    dip_strength = np.linalg.norm(np.array(p_list).reshape(num_syns,3), axis=1)

    zips = []
    for x, z in cell.get_idx_polygons():
        zips.append(list(zip(x, z)))
    zmax = np.max(cell.zend)

    # compute RE_EEG for simulation where all 30 synapses are active at once:
    cell, synapse, electrode_array = simulate(syn_idcs)
    cell.set_pos(x=rz[0], y=rz[1], z=rz[2])
    # compute timepoint with biggest dipole
    dipoles = cell.current_dipole_moment
    timemax = [np.argmax(np.linalg.norm(np.abs(dipoles),axis=1))]
    p = dipoles[timemax]
    # compute LFP with single dipole
    dip_loc = get_dipole_loc(rz, syn_loc)
    lfp_single_dip = fs.calc_potential(p, dip_loc)
    print('pot from single dip computed')
    # compute LFP with multi-dipole
    lfp_multi_dip = fs.calc_potential_from_multi_dipoles(cell, timemax)
    print('pot from multi dip computed')
    # compute relative errors
    RE_30syns = np.abs((lfp_single_dip - lfp_multi_dip)/lfp_multi_dip)
    RE_EEG_30syns = RE_30syns[-1][0]*k_100
    # RE_EEG = np.array(RE_list).reshape(num_syns, num_electrodes)[:,eeg_ind]*k_100
    # RE_ECoG = np.array(RE_list).reshape(num_syns, num_electrodes)[:,ecog_ind]*k_100
    # dip_strength = np.linalg.norm(np.array(p_list).reshape(num_syns,3), axis=1)



    np.savez('./data/compare_multi_single_dipole_segev_syns_from_path',
             lfp_multi = lfp_multi_dip_list,
             lfp_single = lfp_single_dip_list,
             re_eeg = RE_EEG,
             re_ecog = RE_ECoG,
             re = RE_list,
             dipoles = p_list,
             dip_locs = p_loc_list,
             sigmas = sigmas,
             radii = radii,
             synlocs = synlocs,
             zips = zips,
             zmax = zmax,
             RE_EEG_30syns = RE_EEG_30syns,
             zips = zips)

    ################################################################################
    ######################################plot######################################
    ################################################################################

    data = np.load('./data/compare_multi_single_dipole_segev_syns_from_path.npz')
    lfp_multi_dip_list = data['lfp_multi']
    lfp_single_dip_list = data['lfp_single']
    RE_EEG = data['re_eeg']
    RE_ECoG = data['re_ecog']
    p_list = data['dipoles']
    p_loc_list = data['dip_locs']
    sigmas = data['sigmas']
    radii = data['radii']
    synlocs = data['synlocs']
    RE_EEG_30syns = data['RE_EEG_30syns']
    zips = data['zips']

    num_syns = len(synlocs)
    syn_loc_zs = synlocs[:,2] - rz[2]
    dip_strength = np.linalg.norm(np.array(p_list).reshape(num_syns,3), axis=1)
    p_z = np.abs(p_list.reshape(num_syns,3)[:,2])

    RE_list = np.abs((lfp_single_dip_list - lfp_multi_dip_list)/lfp_multi_dip_list)

    plt.close('all')
    fig = plt.figure()
    # line colors
    clrs = plt.cm.viridis(np.linspace(0,0.8,num=num_syns))
    # head color
    head_colors = plt.cm.Pastel1([0,1,2,3])

    # define axes
    ax_setup = plt.subplot2grid((3,4),(0,0))
    ax_pot = plt.subplot2grid((3,4),(1,0), colspan=2)
    ax_pot_RE = plt.subplot2grid((3,4),(2,0), colspan=2)
    ax_RE_EEG = plt.subplot2grid((3,4),(0,2), colspan=2)
    ax_p = plt.subplot2grid((3,4),(1,2), colspan=2)
    ax_RE_EEG_p = plt.subplot2grid((3,4),(2,2), colspan=2)
    # ax_RE_ECoG_p = plt.subplot2grid((4,4),(3,2), colspan=2)

    # plot dipole strength as function of synapse distance from soma
    ax_p.scatter(syn_loc_zs, p_z, s = 5., c = clrs)
    ax_p.set_xlabel(r'synapse distance from soma ($\mu$m)', fontsize=7, labelpad=0.5)
    ax_p.set_ylabel(r'current dipole moment $|p|$ (nA$\mu$m)', fontsize=7)
    # ax_p.set_xticklabels([])

    # plot RE at EEG distance as function of synapse distance from soma
    ax_RE_EEG.scatter(syn_loc_zs, RE_EEG, s = 5., c = clrs)
    ax_RE_EEG.set_xlabel(r'synapse distance from soma', fontsize=7, labelpad=0.5)
    ax_RE_EEG.set_ylabel(r'RE for EEG (%)', fontsize=7)

    # plot RE at EEG distance as function of dipole strength
    ax_RE_EEG_p.plot(np.arange(16), np.ones(16)*RE_EEG_30syns, 'k--', label="all synapses active simultaneously")
    # ax_RE_EEG_p.scatter(dip_strength, RE_EEG, s = 5., c = clrs)
    ax_RE_EEG_p.scatter(p_z, RE_EEG, s = 5., c = clrs)
    ax_RE_EEG_p.set_xlabel(r'current dipole moment $|p|$ (nA$\mu$m)', fontsize=7)
    ax_RE_EEG_p.set_ylabel(r'RE for EEG (%)', fontsize=7)
    ax_RE_EEG_p.legend(loc=1, fontsize=6, frameon=False)


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
    zoom_ax = zoomed_inset_axes(ax_setup, 200, loc=9, bbox_to_anchor=(900, 1800)) # zoom = 6
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

    plt.axis('tight')
    ax_setup.axis('off')
    ax_setup.set_aspect('equal')

    electrode_locs_z = electrode_locs[:,2] - np.max(cell.zend)
    # syns_to_plot = np.arange(0, num_syns, 5)
    syns_to_plot = np.array([5, 25])
    k = 1e3 # from mV to uV
    k1 = 1e-3 # from mum to mm
    electrode_locs_z = electrode_locs_z*k1

    for syn in range(num_syns):
        zoom_ax.plot(synlocs[syn][0], synlocs[syn][2], 'o', color=clrs[syn], ms = .5)

    k_100 = 100
    for i in syns_to_plot:
        # plot lfps
        lfp_single_dip = lfp_single_dip_list[i].reshape(electrode_locs_z.shape)*k
        lfp_multi_dip = lfp_multi_dip_list[i].reshape(electrode_locs_z.shape)*k
        ax_pot.loglog(electrode_locs_z, np.abs(lfp_single_dip), color=clrs[i], linewidth=1.)
        ax_pot.loglog(electrode_locs_z, np.abs(lfp_multi_dip), '--', color=clrs[i], linewidth=1.)

        # plot relative errors
        RE = RE_list[i].reshape(electrode_locs_z.shape)*k_100
        ax_pot_RE.semilogx(electrode_locs_z, RE, color = clrs[i], label=str(i), linewidth=1.)
        zoom_ax.plot(synlocs[i][0], synlocs[i][2], 'o', color=clrs[i], ms = 3)
    ax_pot.plot(100, 0, 'k--', label='multi-dipole')
    ax_pot.plot(100, 0, 'k-', label='single-dipole')
    ax_pot.legend(loc=1, fontsize=6, frameon=False)


    # fix axes
    layer_dist_from_neuron = [(r - np.max(cell.zend))*k1 for r in radii]
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
    ax_pot.set_ylim([1e-6, 1e-2])
    ax_pot_RE.set_ylim([0, 100])
    ax_pot.set_ylabel(r'electric potential $|\Phi|$ ($\mu$V)', fontsize=7)
    ax_pot_RE.set_ylabel(r'RE (%)', fontsize=7)
    ax_pot_RE.set_xlabel(r'distance from top of neuron to electrode (mm)', fontsize=7)
    # mark ECoG and EEG locations
    plt.text(0.161, 0.605, 'ECoG', fontsize=8, transform=plt.gcf().transFigure)
    plt.text(0.467, 0.605, 'EEG', fontsize=8, transform=plt.gcf().transFigure)

    for ax in [ax_p, ax_RE_EEG]:
        ax.set_xlim([-20, 800])

    for ax in [ax_RE_EEG, ax_RE_EEG_p]:
        ax.set_ylim([0, 10])
    ax_p.set_ylim([0, 15])

    ax_RE_EEG_p.set_xlim([0, 15])

    for ax in [ax_pot, ax_pot_RE, ax_p, ax_RE_EEG, ax_RE_EEG_p]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    fig.tight_layout(pad=0.5, h_pad=-1.2, w_pad=1.1)
    fig.set_size_inches(8., 6.)
    fig.subplots_adjust(bottom=.08, top=.91)
    plotting_convention.mark_subplots(fig.axes[:-1], xpos=-0.25)
    plt.savefig('./figures/fig_compare_multi_single_dipole_segev_syns_from_path.png', dpi=300)
