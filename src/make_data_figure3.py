import matplotlib
matplotlib.use("AGG")
import numpy as np
import matplotlib.pyplot as plt
import plotting_convention as plotting_convention
import LFPy
import matplotlib
from matplotlib import colorbar
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import neuron
import os
from os.path import join

def make_data(morphology, cell_model, rot, rz, radii, sigmas,
              electrode_locs, syn_idcs, spiking, syn_input_time):
    # set cell and synapse parameters
    cell_parameters, synapse_parameters = set_parameters(morphology,
                                                         cell_model, spiking)
    [xrot, yrot, zrot] = rot
    # create four-sphere class instance
    fs = LFPy.FourSphereVolumeConductor(radii, sigmas, electrode_locs)
    # lists for storing data:
    p_list = []
    t_max_list = []
    p_loc_list = []
    lfp_single_dip_list = []
    lfp_multi_dip_list = []
    RE_list = []
    synlocs = []
    pz_traces = []
    # get data from num_syns simulations
    num_syns = len(syn_idcs)
    for j in range(num_syns):

        cell = create_cell(cell_parameters, active=spiking,
                           x_rot=xrot, y_rot=yrot, z_rot=zrot)
        ## if you know synidx:
        syn_idx = syn_idcs[j]
        # if you only know synapse location:
        # syn_loc = syn_idcs[j]
        # syn_idx = cell.get_closest_idx(x=syn_loc[0], y=syn_loc[1], z=syn_loc[2])
#        print('syn_idx:', syn_idx)
        cell, synapse, electrode_array = simulate(cell, synapse_parameters, [syn_idx], syn_input_time)
        # print('cell simulated')

        cell.set_pos(x=rz[0], y=rz[1], z=rz[2])
        syn_loc = (cell.xmid[syn_idx], cell.ymid[syn_idx], cell.zmid[syn_idx])
        synlocs.append((syn_loc[0], syn_loc[1], syn_loc[2]))

        # compute current dipole moment and subtract DC-component
        dipoles = cell.current_dipole_moment
        input_idx_before_input = np.argmin(np.abs(cell.tvec - syn_input_time)) - 1
        p_dc = dipoles[input_idx_before_input]
        dipoles -= p_dc

        pz_traces.append(dipoles[:,2])
        # compute timepoint with biggest dipole
        timemax = [np.argmax(np.linalg.norm(np.abs(dipoles),axis=1))]
        t_max_list.append(timemax)
        p = dipoles[timemax]
        # compute LFP with single dipole
        # dip_loc = get_mass_center(cell, timemax)
        dip_loc = get_dipole_loc(rz, syn_loc)
        lfp_single_dip = fs.calc_potential(p, dip_loc)
        # print('pot from single dip computed')
        # compute LFP with multi-dipole

        # subtract DC-component
        multi_p_319, multi_p_locs = cell.get_multi_current_dipole_moments([input_idx_before_input])
        multi_p_dc = multi_p_319
        multi_p, multi_p_locs = cell.get_multi_current_dipole_moments(timemax)
        multi_p -= multi_p_dc

        Ni, Nt, Nd = multi_p.shape
        lfp_multi_dip = np.zeros((len(electrode_locs), Nt))
        for num in range(Ni):
            pot = fs.calc_potential(multi_p[num], multi_p_locs[num])
            lfp_multi_dip += pot

        # print('pot from multi dip computed')
        # compute relative errors
        RE = np.abs((lfp_single_dip - lfp_multi_dip)/lfp_multi_dip)
        print('syn number: {}; syn dist from soma: {}; RE_EEG: {}'.format(syn_idx, syn_loc[2] - rz[2], RE[-1]))
        #add data to lists for storage
        p_list.append(p)
        p_loc_list.append(dip_loc)
        lfp_single_dip_list.append(lfp_single_dip)
        lfp_multi_dip_list.append(lfp_multi_dip)
        RE_list.append(RE)

    # Do this for only one cell for plotting
    zips = []
    for x, z in cell.get_idx_polygons():
        zips.append(list(zip(x, z)))
    zmax = np.max(cell.zend)
    soma_vmem = cell.vmem[0]
    tvec = cell.tvec

    return (p_list, pz_traces, lfp_multi_dip_list, lfp_single_dip_list,
            synlocs, zips, zmax, tvec, soma_vmem)

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
        #print(curr_sec.name())
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

def set_parameters(morphology, cell_model=None, spiking=False):
    """set cell, synapse and electrode parameters"""
    model_folder = join("cell_models", "EyalEtAl2018")
    morph_path = join(model_folder, "Morphs", morphology)

    if cell_model == None:
        cell_parameters = {
            'v_init': -70,
            'morphology': morph_path,
            'passive_parameters': {'g_pas' : 1./30000, 'e_pas' : -70}, # S/cm^2, mV
            'Ra' : 150, # Ω cm
            'cm' : 1, # µF/cm^2
            'nsegs_method': "lambda_f",
            "lambda_f": 100,
            'dt': 2**-4,  # [ms] Should be a power of 2
            'tstart': -10,  # [ms] Simulation start time
            'tstop': 100,  # [ms] Simulation end time
            "pt3d": True,
            'passive': True
            }

    else:
        mod_folder = join(model_folder, 'ActiveMechanisms')
        model_path = join(model_folder, "ActiveModels", cell_model + '_mod')
        neuron.load_mechanisms(mod_folder)

        cell_parameters = {
                'morphology': morph_path,
                'templatefile': model_path + '.hoc',
                'templatename': cell_model,
                'templateargs': morph_path,
                'v_init': -86,
                'passive': False,
                'dt': 2**-4,  # [ms] Should be a power of 2
                'tstart': -200,  # [ms] Simulation start time
                'tstop': 100,  # [ms] Simulation end time
                "pt3d": True,
                'nsegs_method': "lambda_f",
                "lambda_f": 100,
        }


    synapse_parameters = {'e': 0.,  # reversal potential
                      'weight': 0.002 if not spiking else 0.05,  # synapse weight
                      'record_current': True,  # record synapse current
                      # parameters not included in first version of fig
                      'syntype': 'Exp2Syn',
                      'tau1': 1.,  #Time constant, rise
                      'tau2': 3.,  #Time constant, decay
                          }
    return cell_parameters, synapse_parameters


def create_cell(cell_parameters, active=False, x_rot=0, y_rot=0, z_rot=0):
    # create cell with parameters in dictionary
    if not active:
        cell = LFPy.Cell(**cell_parameters)
    else:
        cell = LFPy.TemplateCell(**cell_parameters)

    cell.set_rotation(x=x_rot)
    cell.set_rotation(y=y_rot)
    cell.set_rotation(z=z_rot)

    return cell


def simulate(cell, synapse_parameters, synidx, input_time=20.):
    """set synapse location. simulate cell, synapse and electrodes for input synapse location"""

    for idx in synidx:
        synapse_parameters['idx'] = idx #
        # create synapse with parameters in dictionary
        synapse = LFPy.Synapse(cell, **synapse_parameters)
        synapse.set_spike_times(np.array([input_time]))

    cell.simulate(rec_imem=True, rec_vmem=True, rec_current_dipole_moment=True)

    #create grid electrodes
    electrode_array = LFPy.RecExtElectrode(cell, **electrodeParams)
    electrode_array.calc_lfp()

    return cell, synapse, electrode_array

def get_dipole_loc(rz, syn_loc):
    # mid_pos = np.array([syn_loc[0], syn_loc[1], syn_loc[2]])
    dip_loc = (rz + syn_loc) / 2
    return dip_loc

def get_mass_center(cell, timepoint):
    dips, dip_locs = cell.get_multi_current_dipole_moments(np.array(timepoint))
    masses = np.linalg.norm(dips, axis=2)
    print('masses', masses, masses.shape)
    print('dips', dips, dips.shape)
    print('dip_locs', dip_locs, dip_locs.shape)
    mass_center = 1. / np.sum(masses) * np.sum(np.dot(masses.T, dip_locs), axis=0)
    return mass_center

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
        axis.text(-430, 400, r'$1 \mathsf{mm}$', size = 8, va='center',
                  ha='center', rotation = 'vertical')
    plt.axis('tight')


if __name__ == '__main__':

    # make 4S-parameters
    sigmas = [0.276, 1.65, 0.01, 0.465]
    radii = [89000., 90000., 95000., 100000.]

    spiking = False

    syn_input_time = 20.

    # make electrode array params
    num_electrodes = 40
    electrode_loc_zs = list(np.linspace(radii[0] - 100, radii[-1], num_electrodes))
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

    # cell parameters
    morphology = '2013_03_06_cell03_789_H41_03.ASC'
    rz = np.array([0., 0., 88015.])
    rot = [-np.pi/2, -np.pi/7, 0]

    if spiking:
        cell_model = 'cell0603_03_model_476'
        syn_idcs = [0]
        filename = './data/data_fig3_spiking'
    else:
        cell_model = None
        syn_idcs = [0,338,340,342,344,346,348,349,350,351,352,353,354,
                    355,469,470,471,472,473,563,564,565,566,567,
                    568,569,570,571,572,573,574,575,576,577,578,
                    579,580,586,587,588,589,590]
        filename = './data/data_fig3'

    (p_list, pz_traces, lfp_multi, lfp_single, synlocs, zips,
     zmax, tvec, soma_vmem) = make_data(morphology, cell_model, rot, rz, radii,
                                        sigmas, electrode_locs, syn_idcs,
                                        spiking, syn_input_time)


    np.savez(filename,
             lfp_multi = lfp_multi,
             lfp_single = lfp_single,
             dipoles = p_list,
             synlocs = synlocs,
             zips = zips,
             zmax = zmax,
             rz = rz,
             electrode_locs = electrode_locs,
             tvec = tvec,
             pz_traces = pz_traces,
             soma_vmem = soma_vmem,
             sigmas=sigmas,
             radii=radii,
             syn_idcs=syn_idcs)
