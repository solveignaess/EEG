# Show px, py, pz and EEG from simulation of various neuron types with different input patterns

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

s = 6
random.seed(s)

def make_cell(morphology, morph_type='l23'):
    """set synapse location. simulate cell, synapse and electrodes for input synapse location"""

    cell_parameters = {
        'v_init': -70,
        'morphology': morphology,
        'passive_parameters': dict(e_pas=-70),
        'nsegs_method': "lambda_f",
        "lambda_f": 100,
        'dt': 2**-4,  # [ms] Should be a power of 2
        'tstart': 0,  # [ms] Simulation start time
        'tstop': 200,  # [ms] Simulation end time
        "pt3d": True,
        # 'custom_code': custom_code
        }

    # create cell with parameters in dictionary
    print('initiate cell')
    cell = LFPy.Cell(**cell_parameters)
    cell.set_rotation(x=np.pi/2)

    if morph_type == 'l23':
        soma_top_dist = np.max(cell.zend)
        soma_brain_surf_dist = soma_top_dist + 10.
        cell.set_pos(x = 0., y = 0., z = 378.)


    return cell

def make_synapse(cell, weight, input_idx, input_spike_train, e=0.):
    synapse_parameters = {
        'idx': input_idx,
        'e': e,
        'syntype': 'Exp2Syn',
        # 'tau': 2.,
        'tau1' : 1.,                #Time constant, rise
        'tau2' : 3.,                #Time constant, decay
        # 'tau2': 2,
        'weight': weight,
        'record_current': False,
    }
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(input_spike_train)
    return cell, synapse

def make_input(cell, weight=0.005):
    # apical exc input at time=10
    input_idxs = cell.get_rand_idx_area_norm(section='allsec', nidx=100, z_min=np.max(cell.zmid)-200)
    pulse_center = 10 #+ np.random.normal(0, 1)
    for input_idx in input_idxs:
        input_spike_train = np.random.normal(pulse_center, 3, size=1)
        cell, synapse = make_synapse(cell, weight/100. * np.random.normal(1., 0.2), input_idx, input_spike_train)

    # basal exc input at time=60
    input_idxs = cell.get_rand_idx_area_norm(section='allsec', nidx=100,
                                             z_min=np.min(cell.zmid), z_max=cell.zmid[0]+100)
    pulse_center = 60 #+ np.random.normal(0, 1)
    for input_idx in input_idxs:
        input_spike_train = np.random.normal(pulse_center, 3, size=1)
        cell, synapse = make_synapse(cell, weight/100. * np.random.normal(1., 0.2), input_idx, input_spike_train)

    # overall exc input at t=110 and t=160
    input_idxs = cell.get_rand_idx_area_norm(section='allsec', nidx=400)
    pulse_center = 110 #+ np.random.normal(0, 1)
    for input_idx in input_idxs:
        input_spike_train = np.random.normal(pulse_center, 3, size=1)
        input_spike_train2 = input_spike_train + 50.
        wght = weight/200. * np.random.normal(1., 0.2)
        cell, synapse = make_synapse(cell, wght, input_idx, input_spike_train)
        cell2, synapse2 = make_synapse(cell, wght, input_idx, input_spike_train2)

    # basal inh input at t=160
    input_idxs = cell.get_rand_idx_area_norm(section='allsec', nidx=50,
                                             z_min=np.min(cell.zmid), z_max=cell.zmid[0]+100)
    pulse_center = 160 #+ np.random.normal(0, 1)
    for input_idx in input_idxs:
        input_spike_train = np.random.normal(pulse_center, 3, size=1)
        cell, synapse = make_synapse(cell, 4*weight/100. * np.random.normal(1., 0.2),
                                     input_idx, input_spike_train, e=-90.)
    return cell, synapse

def return_head_parameters():
    radii = [79000., 80000., 85000., 90000.]
    sigmas = [0.3, 1.5, 0.015, 0.3]
    rad_tol = 1e-2
    return radii, sigmas, rad_tol

def return_measurement_coords(radii, rad_tol):
    theta_step = 0.2
    phi_step = 60
    theta, phi_angle = np.mgrid[0:0.8:theta_step, 0:360+phi_step:phi_step]
    theta = theta.flatten()
    phi_angle = phi_angle.flatten()

    theta_r = np.deg2rad(theta)
    phi_angle_r = np.deg2rad(phi_angle)

    rad_tol = 1e-2
    scalp_rad = radii[-1]

    x_eeg = (scalp_rad - rad_tol) * np.sin(theta_r) * np.cos(phi_angle_r)
    y_eeg = (scalp_rad - rad_tol) * np.sin(theta_r) * np.sin(phi_angle_r)
    z_eeg = (scalp_rad - rad_tol) * np.cos(theta_r)
    eeg_coords = np.vstack((x_eeg, y_eeg, z_eeg)).T

    return eeg_coords

def return_eeg(cell, radii, sigmas, eeg_coords):
    # compute current dipole moment
    P = cell.current_dipole_moment
    # set dipole position
    r_soma_syns = [cell.get_intersegment_vector(idx0=0, idx1=i)
                   for i in cell.synidx]
    r_mid = np.average(r_soma_syns, axis=0)
    somapos = np.array([0., 0., 77500])
    dipole_pos = r_mid + somapos
    # compute eeg with 4S model
    fs_eeg = LFPy.FourSphereVolumeConductor(radii, sigmas, eeg_coords)
    eeg = fs_eeg.calc_potential(P, dipole_pos)
    eeg_avg = np.average(eeg, axis=0)

    # convert from mV to pV:
    eeg_avg = eeg_avg*1e9

    return eeg_avg

def plot_neuron(axis, cell, clr, lengthbar=False, celltype='l23'):
    shift = 0.
    if celltype == 'l23':
        shift = 700.
    elif celltype == 'l5i':
        shift = 1400.
    [axis.plot([cell.xstart[idx] + shift, cell.xend[idx] + shift],
                   [cell.zstart[idx], cell.zend[idx]], c=clr, clip_on=False)
                   for idx in range(cell.totnsegs)]
    axis.plot(cell.xmid[0] + shift, cell.zmid[0], 'o', c="k", ms=0.001)
    plt.axis('tight')
    # axis.axis('off')
    axis.set_aspect('equal')

if __name__ == '__main__':

    sim_cells = True
    compute_eegs = True
    morphology_dict = {'l23' : './cell_models/segev/CNG_version/2013_03_06_cell03_789_H41_03.CNG.swc',
                       'l5' : './cell_models/hay/L5bPCmodelsEH/morphologies/cell1.asc',
                       'l5i' : './cell_models/L5_ChC_cNAC187_3/morphology/rp110201_L5-1_idA_-_Scale_x1.000_y0.975_z1.000_-_Clone_3_no_axon.asc'} # chandelier cell



    model_folder = None
    # simulate cell
    if sim_cells:
        cells = []
        for m in morphology_dict.keys():
            print('morph_type', m)
            morphology = morphology_dict[m]
            cell = make_cell(morphology, morph_type=m)
            # if m == 'l5i':
            #     forall delete_section('axon')
            cell,synapse = make_input(cell)
            cell.simulate(rec_vmem=True, rec_current_dipole_moment=True)
            cells.append(cell)

    # set 4S parameters
    if compute_eegs:
        radii, sigmas, rad_tol = return_head_parameters()
        eeg_coords = return_measurement_coords(radii, rad_tol)
        eeg = True

        if eeg:
            # compute EEG with 4s
            eeg_segev = return_eeg(cells[0], radii, sigmas, eeg_coords)
            eeg_hay = return_eeg(cells[1], radii, sigmas, eeg_coords)
            eeg_inter = return_eeg(cells[2], radii, sigmas, eeg_coords)
        eegs = [eeg_segev, eeg_hay, eeg_inter]

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

    num_cells = len(morphology_dict)
    clrs = plt.cm.viridis(np.linspace(0,0.8,num=num_cells))
    clrs2 = plt.cm.viridis(np.linspace(0.03,0.83,num=num_cells))

    ax_syns.plot(cells[0].sptimeslist[0], cells[0].zmid[cells[0].synapses[0].idx], 'o',
                 ms = 3, markerfacecolor = 'None', markeredgecolor='gray', markeredgewidth=.3, label='excitatory')
    ax_syns.plot(cells[0].sptimeslist[0], cells[0].zmid[cells[0].synapses[0].idx], 'x',
                 ms = 3., markerfacecolor = 'gray', markeredgecolor='gray', markeredgewidth=1., label='inhibitory')
    for cellnum in range(num_cells):
        cell = cells[cellnum]
        for syn_number, syn in enumerate(cell.synapses):
            ec = clrs[cellnum]
            m = '.' if syn.kwargs['e'] > -60 else 'x'
            mew = .3 if syn.kwargs['e'] > -60 else .7
            ax_syns.plot(cell.sptimeslist[syn_number],
                       np.ones(len(cell.sptimeslist[syn_number])) * cell.zmid[syn.idx],
                       marker=m, ms = 3., markerfacecolor='None', #alpha=a,
                       markeredgecolor=ec, markeredgewidth = mew)
        p = cell.current_dipole_moment
        ax_p.plot(cell.tvec, p[:,0] + 300, color=clrs[cellnum])
        ax_p.plot(cell.tvec, p[:,1] + 150, color=clrs[cellnum])
        ax_p.plot(cell.tvec, p[:,2], color=clrs[cellnum])
        ax_eeg.plot(cell.tvec, eegs[cellnum], color=clrs[cellnum])

    for i in range(num_cells):
        cell = cells[i]
        plot_neuron(ax_morph, cell, clrs[i], celltype=list(morphology_dict)[i])
        ax_morph.plot(0,0,'o', ms = 1e-4)
    morph_names = ['hay l5', 'segev l23', 'inter l5']

    ax_morph.set_title('morphologies')
    ax_syns.set_title('synaptic input')
    ax_p.set_title('current dipole moment')
    # ax_px.set_title('current dipole moment')
    ax_eeg.set_title('EEG')

    ax_morph.spines['top'].set_visible(False)
    ax_morph.spines['right'].set_visible(False)
    ax_morph.spines['bottom'].set_visible(False)
    ax_morph.set_xticks([])
    ax_morph.set_xticklabels([])
    ax_morph.set_ylabel(r'z ($\mu$m)')

    for ax in [ax_syns, ax_p, ax_eeg]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim([0, 200])
    ax_syns.set_ylabel(r'z ($\mu$m)')
    ax_syns.legend(loc=1, bbox_to_anchor=(1.1, 0.8, 0.1, 0.1), fontsize='x-small', frameon=False)

    for ax in [ax_morph, ax_syns]:
        ax.set_ylim([-250, 1250])
    ax_eeg.set_ylim([-96, 50])
    ax_eeg.set_yticks([-75, -50, -25., 0., 25., 50.])

    end_point = cell.tvec[-1]+ 1.
    start_point = -20
    dx = .5
    ax_p.text(end_point+dx, -12, r'10 nA$\mu$m', fontsize='small')
    ax_p.plot([end_point, end_point], [-5.3, -5.], 'k-', lw=3.)
    # points for testing actual length of length bar
    # ax_p.plot(end_point, 0, 'ro', ms=1.)
    # ax_p.plot(end_point, -10, 'ro', ms=1.)
    # ax_p.plot([end_point+2., end_point+2.], [-10, 0], 'k-', lw=5.)


    ax_p.text(start_point, 0, r'$p_z$')
    ax_p.text(start_point, 150, r'$p_y$')
    ax_p.text(start_point, 300, r'$p_x$')
    ax_p.spines['left'].set_visible(False)
    ax_p.set_yticks([])
    ax_p.set_yticklabels([])

    ax_eeg.set_xlabel('time (ms)')

    ax_eeg.set_ylabel(r'$\Phi$ (mV)')
    for ax in [ax_p, ax_eeg, ax_syns]:
        ax.set_xlabel(r'time (ms)')

    fig.tight_layout(pad=0.5, h_pad=1., w_pad=.7)
    title = './figures/fig_compare_neurons_l5ChC.png'
    plt.savefig(title, dpi=600) #_nbc
