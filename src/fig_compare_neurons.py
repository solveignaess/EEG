# Show px, py, pz and EEG from simulation of various neuron types with different input patterns
import os
import matplotlib
matplotlib.use("AGG")
import numpy as np
import matplotlib.pyplot as plt
import plotting_convention as plotting_convention
import LFPy
import neuron
from neuron import h
from matplotlib import colorbar
from matplotlib.collections import PolyCollection
import time
import random
from os.path import join

mod_folder = join(os.path.dirname(LFPy.__file__), "test")
neuron.load_mechanisms(join(mod_folder))

def make_cell(morphology, morph_type='l23'):
    """set synapse location. simulate cell, synapse and electrodes for input synapse location"""

    cell_parameters = {
        'v_init': -70,
        'morphology': morphology,
        'passive_parameters': {'g_pas' : 1./30000, 'e_pas' : -70}, # S/cm^2, mV
        'Ra' : 150, # Ω cm
        'cm' : 1, # µF/cm^2
        'nsegs_method': "lambda_f",
        "lambda_f": 100,
        'dt': 2**-4,  # [ms] Should be a power of 2
        'tstart': -10,  # [ms] Simulation start time
        'tstop': 250,  # [ms] Simulation end time
        "pt3d": True,
        'passive': True
        }

    # create cell with parameters in dictionary
    print('initiate cell')
    cell = LFPy.Cell(**cell_parameters)

    # rotate cell
    if morph_type == 'l23':
        cell.set_rotation(x=-np.pi/2)
        cell.set_rotation(y=-np.pi/7)
    else:
        cell.set_rotation(x=np.pi/2)


    return cell

def make_synapse(cell, weight, input_idx, input_spike_train, e=0., syntype='Exp2Syn'):
    synapse_parameters = {
        'idx': input_idx,
        'e': e,
        'syntype': syntype,
        'tau1' : 1.,                   #Time constant, rise
        'tau2' : 3., #5                #Time constant, decay
        'weight': weight,
        'record_current': False,
    }

    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([input_spike_train]))
    return cell, synapse

def make_input(cell, weight=0.01, syntype='Exp2Syn', morph_type='l23'):

    if morph_type == 'l5i':
        zmin_apical = np.min(cell.zmid)
        zmax_basal = np.max(cell.zmid)
    else:
        zmin_apical = np.max(cell.zmid)-200
        zmax_basal = cell.zmid[0]+100

    # apical exc input at time=50
    num_apic = 100
    pulse_center = 50
    weight_apic = weight / num_apic
    input_idxs = cell.get_rand_idx_area_norm(section='allsec', nidx=num_apic,
                                             z_min=zmin_apical)
    input_spike_train = np.random.normal(pulse_center, 3, size=num_apic)
    for n, input_idx in enumerate(input_idxs):
        cell, synapse = make_synapse(cell, weight_apic, input_idx,
                                     input_spike_train[n], syntype=syntype)

    # basal exc input at time=100
    num_basal = 100
    pulse_center = 100
    input_idxs = cell.get_rand_idx_area_norm(section='allsec', nidx=num_basal,
                                             z_min=np.min(cell.zmid), z_max=zmax_basal)
    weight_basal = weight / num_basal
    input_spike_train = np.random.normal(pulse_center, 3, size=num_basal)
    for n, input_idx in enumerate(input_idxs):
        cell, synapse = make_synapse(cell, weight_basal, input_idx,
                                     input_spike_train[n], syntype=syntype)

    # overall exc input at t=150
    num_uniform = 400
    weight_uniform = weight / num_uniform
    input_idxs = cell.get_rand_idx_area_norm(section='allsec', nidx=num_uniform)
    pulse_center = 150
    input_spike_train = np.random.normal(pulse_center, 3, size=num_uniform)
    for n, input_idx in enumerate(input_idxs):
        cell, synapse = make_synapse(cell, weight_uniform, input_idx,
                                     input_spike_train[n], syntype=syntype)

    # overall exc input for t=200, same input_idxs as above
    pulse_center = 200
    for n, input_idx in enumerate(input_idxs):
        cell, synapse = make_synapse(cell, weight_uniform, input_idx,
                                     input_spike_train[n] + 50, syntype=syntype)

    # basal inh input at t=200
    num_inhib = 50
    weight_inhib = weight / num_inhib
    input_idxs = cell.get_rand_idx_area_norm(section='allsec', nidx=num_inhib,
                                             z_min=np.min(cell.zmid), z_max=zmax_basal)
    input_spike_train = np.random.normal(pulse_center, 3, size=num_inhib)
    for n, input_idx in enumerate(input_idxs):

        cell, synapse = make_synapse(cell, weight_inhib,
                                     input_idx, input_spike_train[n],
                                     e=-90., syntype=syntype)
    return cell, synapse

def return_head_parameters():
    # From Huang et al. (2013): 10.1088/1741-2560/10/6/066004
    sigmas = [0.276, 1.65, 0.01, 0.465]
    radii = [89000., 90000., 95000., 100000.]
    rad_tol = 1e-2
    return radii, sigmas, rad_tol


def return_measurement_coords(radii, rad_tol):
    rad_tol = 1e-2
    scalp_rad = radii[-1]
    return np.array([[0, 0, scalp_rad - rad_tol]])

def return_eeg(cell, radii, sigmas, eeg_coords, morph_type):
    # compute current dipole moment
    P = cell.current_dipole_moment
    # set dipole position
    r_soma_syns = [cell.get_intersegment_vector(idx0=0, idx1=i)
                   for i in cell.synidx]
    r_mid = np.average(r_soma_syns, axis=0)
    if morph_type == 'l5i':
        somapos = np.array([0., 0., radii[0]-1200])
    else:
        somapos = np.array([0., 0., radii[0] - np.max(cell.zend) - 50])
    dipole_pos = r_mid + somapos
    cell.set_pos(x=somapos[0], y=somapos[1], z=somapos[2])

    # multi_dipoles, dipole_locs = cell.get_multi_current_dipole_moments()
    # t_point = 800
    # P_from_multi_dipoles = np.sum(multi_dipoles[:,t_point,:],axis=0)
    # print('single dipole:', P[t_point])
    # print('sum of multi-dipoles', P_from_multi_dipoles)
    # compute eeg with 4S model
    fs_eeg = LFPy.FourSphereVolumeConductor(radii, sigmas, eeg_coords)
    eeg = fs_eeg.calc_potential(P, dipole_pos)
    eeg_multidip = fs_eeg.calc_potential_from_multi_dipoles(cell)

    eeg_avg = np.average(eeg, axis=0)
    eeg_multidip_avg = np.average(eeg_multidip, axis=0)
    # convert from mV to pV:
    eeg_avg = eeg_avg * 1e9
    eeg_multidip_avg = eeg_multidip_avg * 1e9

    return eeg_avg, eeg_multidip_avg

def plot_neuron(axis, cell, clr, lengthbar=False, celltype='l23'):
    shift = 0.
    if celltype == 'l5':
        shift = 750.
    elif celltype == 'l5i':
        shift = 1400.
    [axis.plot([cell.xstart[idx] + shift, cell.xend[idx] + shift],
                   [cell.zstart[idx], cell.zend[idx]], c=clr, clip_on=False)
                   for idx in range(cell.totnsegs)]
    # axis.plot(cell.xmid[0] + shift, cell.zmid[0], 'o', c="k", ms=0.001)
    plt.axis('tight')
    # axis.axis('off')
    # axis.set_aspect('equal')

if __name__ == '__main__':
    s = 15
    np.random.seed(seed=s)

    sim_cells = True
    compute_eegs = True
    current_based = False
    morphology_dict = {
       'l23' : './cell_models/segev/CNG_version/2013_03_06_cell03_789_H41_03.ASC',
       'l5' : './cell_models/hay/L5bPCmodelsEH/morphologies/cell1.asc',
       'l5i' : './cell_models/L5_ChC_cNAC187_3/morphology/rp110201_L5-1_idA_-_Scale_x1.000_y0.975_z1.000_-_Clone_3_no_axon.asc'# chandelier cell
       }
    model_folder = None
    # set 4S parameters
    if compute_eegs:
        radii, sigmas, rad_tol = return_head_parameters()
        eeg_coords = return_measurement_coords(radii, rad_tol)
        # eeg = True
    # simulate cell
    cells = []
    eegs = []
    eegs_multidip = []
    error_scaled = []
    if current_based:
        syntype='ExpSynI'
        weight=0.5
    else:
        syntype='Exp2Syn'
        # weight=0.005
        weight=0.01
    for m in morphology_dict.keys():
        print('morph_type', m)
        morphology = morphology_dict[m]
        cell = make_cell(morphology, morph_type=m)

        cell, synapse = make_input(cell, weight=weight, syntype=syntype, morph_type=m)
        cell.simulate(rec_vmem=True, rec_current_dipole_moment=True, rec_imem=True)
        cells.append(cell)

        # compute EEG with 4s
        eeg, eeg_multidip = return_eeg(cell, radii, sigmas, eeg_coords, m) # units [pV]
        eegs.append(eeg)
        eegs_multidip.append(eeg_multidip)
        ind_max_error = np.argmax(np.abs(eeg_multidip))
        error = np.max(np.abs(eeg - eeg_multidip)/np.abs(eeg_multidip[ind_max_error]))
        print('maximum global error for ', m, 'was', error)
        error_scaled.append(error)

################################################################################
###################################plotting#####################################
################################################################################
    plt.rcParams.update({
        'axes.labelsize' : 8,
        'axes.titlesize' : 8,
        #'figure.titlesize' : 8,
        'font.size' : 8,
        'ytick.labelsize' : 8,
        'xtick.labelsize' : 8,
        'lines.linewidth' : 1.
    })

    plt.close('all')
    fig = plt.figure()
    ax_morph = plt.subplot2grid((2,2),(0,0))
    ax_syns = plt.subplot2grid((2,2),(0,1))
    ax_p = plt.subplot2grid((2,2),(1,0))
    ax_eeg = plt.subplot2grid((2,2), (1,1))

    morph_ylim = [radii[0] - 1500, radii[0]]

    num_cells = len(morphology_dict)
    # clrs = plt.cm.Accent(np.linspace(0, 0.8, num=num_cells))
    clrs = ['b', 'orangered', 'orange']

    ax_syns.vlines([50, 100, 150, 200], radii[0]-1500, radii[0] - 50, colors='k',
                   linestyles='-', linewidth=0.5)

    ax_syns.plot(-cells[0].sptimeslist[0], cells[0].zmid[cells[0].synapses[0].idx], '.',
                 ms = 1, markerfacecolor = 'gray', markeredgecolor='gray',
                 label='excitatory')

    ax_syns.plot(-cells[0].sptimeslist[0], cells[0].zmid[cells[0].synapses[0].idx], 'o',
                 ms = 3., markerfacecolor = 'gray', markeredgecolor='k',
                 label='inhibitory')

    ax_eeg.vlines([50, 100, 150, 200], -96, 41, colors='k', linestyles='-', linewidth=.5)
    ax_eeg.plot([300, 310], [0, 10], 'gray', label='single-dipole')
    ax_eeg.plot([300, 310], [0, 10], 'gray',ls=':', label='multi-dipole')

    ax_p.vlines([50, 100, 150, 200], -110, 301, colors='k', linestyles='-', linewidth=.5)

    for cellnum in range(num_cells):
        cell = cells[cellnum]
        for syn_number, syn in enumerate(cell.synapses):
            ec = clrs[cellnum]
            m = '.' if syn.kwargs['e'] > -60 else 'o'
            mec = ec if syn.kwargs['e'] > -60 else 'k'
            # mew =
            ms = 1 if syn.kwargs['e'] > -60 else 2.5
            zorder = -1 if syn.kwargs['e'] > -60 else 1  # Plot crosses in front for visibility

            ax_syns.plot(cell.sptimeslist[syn_number],
                       np.ones(len(cell.sptimeslist[syn_number])) * cell.zmid[syn.idx],
                       marker=m, ms = ms, mec=mec, zorder=zorder, c=ec)

        p = cell.current_dipole_moment
        ax_p.plot(cell.tvec, p[:,0] + 100, color=clrs[cellnum], lw=1.5)
        ax_p.plot(cell.tvec, p[:,1] + 50, color=clrs[cellnum], lw=1.5)
        ax_p.plot(cell.tvec, p[:,2], color=clrs[cellnum], lw=1.5)
        print('morph:', list(morphology_dict.keys())[cellnum])
        # print('min pz:', np.min(p[:,2]))
        # print('max px:', np.max(p[:,0]+100))
        ax_eeg.plot(cell.tvec, eegs[cellnum], color=clrs[cellnum], clip_on=False, lw=1.5)
        ax_eeg.plot(cell.tvec, eegs_multidip[cellnum], ':', color='gray', lw=1.5)
        print('max eeg:', np.max(eegs[cellnum]))

    ax_syns.text(35, radii[0] + 20, 'apical', fontsize=7)
    ax_syns.text(85, radii[0] + 20, 'basal', fontsize=7)
    ax_syns.text(130, radii[0] + 20, 'uniform', fontsize=7)
    ax_syns.text(185, radii[0] + 50, 'uniform', fontsize=7)
    ax_syns.text(185, radii[0] -40, '+ basal inhib.', fontsize=7)

    for i in range(num_cells):
        cell = cells[i]
        plot_neuron(ax_morph, cell, clrs[i], celltype=list(morphology_dict)[i])
        # ax_morph.plot(0,0,'o', ms = 1e-4)
    morph_names = ['hay l5', 'segev l23', 'inter l5']

    ax_morph.set_title('morphologies', pad=11)
    ax_syns.set_title('synaptic input', pad=11)
    ax_p.set_title('current dipole moment', pad=11)
    # ax_px.set_title('current dipole moment')
    ax_eeg.set_title('EEG', pad=11)

    ax_morph.spines['top'].set_visible(False)
    ax_morph.spines['right'].set_visible(False)
    ax_morph.spines['bottom'].set_visible(False)
    ax_morph.set_xticks([])
    ax_morph.set_xticklabels([])
    ax_morph.set_ylabel(r'z ($\mu$m)')
    ax_morph.text(-250, radii[0] - 1250, 'human L2/3 PC', fontsize=7.)
    ax_morph.text(570, radii[0] - 1550, 'rat L5 PC', fontsize=7.)
    ax_morph.text(1220, radii[0] - 1500, 'rat L5 ChC', fontsize=7.)

    for ax in [ax_syns, ax_p, ax_eeg]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim([0, 250])
    ax_syns.set_ylabel(r'z ($\mu$m)')
    ax_syns.legend(loc=1, bbox_to_anchor=(1.1, 0.8, 0.1, 0.1), fontsize='small', frameon=False)

    for ax in [ax_morph, ax_syns]:
        ax.set_ylim(morph_ylim)
        ax.set_yticks(np.linspace(morph_ylim[0], morph_ylim[1], num=7))
        ax.set_yticklabels([str(int(i)) for i in np.linspace(-1500, 0, num = 7)])
    ax_eeg.set_ylim([-40, 30])
    # ax_eeg.set_yticks([-75, -50, -25., 0., 25., 50.])
    ax_eeg.legend(loc=1, bbox_to_anchor=(1.12, 0.8, 0.1, 0.1), fontsize='small', frameon=False)

    # end_point = cell.tvec[-1]+ 1.
    start_point = -20
    dx = .5
    ax_p.text(235, -25, r'50 nA$\mu$m', fontsize='small')
    ax_p.plot([230, 230], [-2., -52.], 'k-')
    # # points for testing actual length of length bar
    # ax_p.plot(end_point, 0, 'ro', ms=1.)
    # ax_p.plot(end_point, -50, 'ro', ms=1.)

    ax_p.text(start_point, 0, r'$p_z$')
    ax_p.text(start_point, 50, r'$p_y$')
    ax_p.text(start_point, 100, r'$p_x$')
    ax_p.spines['left'].set_visible(False)
    ax_p.set_ylim([-60, 110])
    ax_p.set_yticks([])
    ax_p.set_yticklabels([])

    ax_eeg.set_xlabel('time (ms)')

    ax_eeg.set_ylabel(r'$\Phi$ (pV)', labelpad=7.5)
    for ax in [ax_p, ax_eeg, ax_syns]:
        ax.set_xlabel(r'time (ms)')

    # label axes
    xpos = [0.02, 0.522, 0.02, 0.522]
    ypos = [0.978, 0.978, 0.475, 0.475]
    letters = 'ABCD'
    for i in range(len(letters)):
        fig.text(xpos[i], ypos[i], letters[i],
             horizontalalignment='center',
             verticalalignment='center',
             fontweight='demibold',
             fontsize=8)

    fig.tight_layout(pad=0.5, h_pad=1.3, w_pad=.7)
    if current_based:
        title = './figures/figure3_current_based.pdf'
    else:
        # title = './figures/fig_compare_neurons_l5ChC_wmd.png'
        title = './figures/figure3.pdf'
    plt.savefig(title, dpi=600) #_nbc
