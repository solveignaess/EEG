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
from mpi4py import MPI

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

def make_data(morphology, cell_model, rot, rz, radii, sigmas, electrode_locs, syn_idcs, active=False):
    # set cell and synapse parameters
#    print(syn_idcs)
    cell_parameters, synapse_parameters = set_parameters(morphology, cell_model)
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
#    print(syn_idcs)
    # get data from num_syns simulations
    num_syns = len(syn_idcs)
    for j in range(num_syns):
        print('syn number:', j)
        cell = create_cell(cell_parameters, active=active,
                           x_rot=xrot, y_rot=yrot, z_rot=zrot)
        ## if you know synidx:
        syn_idx = syn_idcs[j]
        # if you only know synapse location:
        # syn_loc = syn_idcs[j]
        # syn_idx = cell.get_closest_idx(x=syn_loc[0], y=syn_loc[1], z=syn_loc[2])
#        print('syn_idx:', syn_idx)
        cell, synapse, electrode_array = simulate(cell, synapse_parameters, [syn_idx])
        print('cell simulated')
        syn_loc = (cell.xmid[syn_idx], cell.ymid[syn_idx], cell.zmid[syn_idx])
        synlocs.append((syn_loc[0], syn_loc[1], syn_loc[2]+rz[2]))
        cell.set_pos(x=rz[0], y=rz[1], z=rz[2])

        # compute current dipole moment and subtract DC-component
        dipoles = cell.current_dipole_moment
        p_dc = (dipoles[0] + dipoles[319])/2
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
        print('pot from single dip computed')
        # compute LFP with multi-dipole

        # subtract DC-component
        multi_p_0, multi_p_locs = cell.get_multi_current_dipole_moments([0])
        multi_p_319, multi_p_locs = cell.get_multi_current_dipole_moments([319])
        multi_p_dc = (multi_p_0 + multi_p_319)/2
        multi_p, multi_p_locs = cell.get_multi_current_dipole_moments(timemax)
        multi_p -= multi_p_dc

        Ni, Nt, Nd = multi_p.shape
        lfp_multi_dip = np.zeros((len(electrode_locs), Nt))
        for num in range(Ni):
            pot = fs.calc_potential(multi_p[num], multi_p_locs[num])
            lfp_multi_dip += pot

        print('pot from multi dip computed')
        # compute relative errors
        RE = np.abs((lfp_single_dip - lfp_multi_dip)/lfp_multi_dip)

        #add data to lists for storage
        p_list.append(p)
        p_loc_list.append(dip_loc)
        lfp_single_dip_list.append(lfp_single_dip)
        lfp_multi_dip_list.append(lfp_multi_dip)
        RE_list.append(RE)

        zips = []
        for x, z in cell.get_idx_polygons():
            zips.append(list(zip(x, z)))
        zmax = np.max(cell.zend)
        soma_vmem = cell.vmem[0]
        tvec = cell.tvec

    return p_list, p_loc_list, pz_traces, lfp_multi_dip_list, lfp_single_dip_list, RE_list, synlocs, zips, zmax, tvec, t_max_list, soma_vmem

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

def set_parameters(morphology, cell_model=None):
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


    synapse_parameters = {'e': 0., # reversal potential
                      # 'tau': 5., # synapse time constant (first fig version)
                      'weight': 0.002, # synapse weight
                      'record_current': True, # record synapse current
                      # parameters not included in first version of fig
                      'syntype': 'Exp2Syn',
                      'tau1': 1., #Time constant, rise
                      'tau2': 3., #Time constant, decay
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

def simulate(cell, synapse_parameters, synidx):
    """set synapse location. simulate cell, synapse and electrodes for input synapse location"""

    for idx in synidx:
        synapse_parameters['idx'] = idx #
        # create synapse with parameters in dictionary
        synapse = LFPy.Synapse(cell, **synapse_parameters)
        synapse.set_spike_times(np.array([input_time]))

    cell.simulate(rec_imem = True,
                  rec_vmem = True,
                  rec_current_dipole_moment=True)

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
        axis.text(-430, 400, r'$1 \mathsf{mm}$', size = 8, va='center', ha='center', rotation = 'vertical')
    plt.axis('tight')


if __name__ == '__main__':

    # make 4S-parameters
    sigmas = [0.276, 1.65, 0.01, 0.465]
    radii = [79000., 80000., 85000., 90000.]

    cell_dict = {'cellnames': ['3a',
                               # '3b',
                               # '5',
                               # '6',
                               # '8'#,
                               #, '11'
                               ],
                  'morphs': ['2013_03_06_cell03_789_H41_03.ASC',
                             # '2013_03_13_cell03_1204_H42_02.ASC',
                             # '2013_03_13_cell05_675_H42_04.ASC',
                             # '2013_03_13_cell06_945_H42_05.ASC',
                             # '2013_03_06_cell08_876_H41_05_Cell2.ASC'#,
                             # '2013_03_06_cell11_1125_H41_06.ASC'
                             ],
                  'active_models': [None,#'cell0603_03_model_476', #None
                                    # 'cell1303_03_model_448',
                                    # 'cell1303_05_model_643',
                                    # 'cell1303_06_model_263',
                                    # 'cell0603_08_model_602'#,
                                    # 'cell0603_11_model_937.hoc',
                                    ],
                   'soma_positions': [np.array([0., 0., 88016.]), # From: [0, 0, radii[0] - np.max(cell.zend) - 200]
                           # np.array([0., 0., 77643.]),
                           # np.array([0., 0., 78175.]),
                           # np.array([0., 0., 77888.]),
                           # np.array([0., 0., 77947.])#,
                           # np.array([0., 0., 77747.])
                           ],
                   'syn_idcs': [#[0]
                                  [0,338,340,342,344,346,348,349,350,351,352,353,354,
                                    355,469,470,471,472,473,563,564,565,566,567,
                                    568,569,570,571,572,573,574,575,576,577,578,
                                    579,580,586,587,588,589,590]  # from return_path_to_tip_idcs(cell, pos_x=0, pos_y=0, pos_z=1000)

                                ],
                    'rots': [[-np.pi/2, -np.pi/7, 0],
                             # [np.pi/2, 0., 0.],
                             # [-np.pi/2, -11*np.pi/16, 0.],
                             # [np.pi/2, -np.pi/2, 0.],
                             # [-np.pi/2., np.pi/7, 0.]#,
                             # [-np.pi/2, -np.pi/8, 0.],
                             ]
                    }
    savefolder = "data"

    # sigmas = [0.3, 1.5, 0.015, 0.3]  #
    # radii = [79000., 80000., 85000., 90000.]
    # From Huang et al. (2013): 10.1088/1741-2560/10/6/066004
    sigmas = [0.276, 1.65, 0.01, 0.465]
    radii = [89000., 90000., 95000., 100000.]
>>>>>>> 79b0753e646bc28f88d9118168720ff5ccb6646d

    # active = True
    active = False
    syn_input_time = 20.

    # cell_parameters, synapse_parameters = set_parameters(cell_dict['morphs'][0],
    #                                                      cell_dict['active_models'][0])
    # [xrot, yrot, zrot] = cell_dict['rots'][0]
    # cell = create_cell(cell_parameters, active=active, x_rot=xrot, y_rot=yrot, z_rot=zrot)
    # syn_idxs = return_path_to_tip_idcs(cell, pos_x=0, pos_y=0, pos_z=1000)

    # soma_z_pos =  # Position cell top 200 um below brain surface
    # soma_pos = [0, 0, radii[0] - np.max(cell.zend) - 200]
    # print(soma_pos)
    # sys.exit()

    # make electrode array params
    num_electrodes = 40
    electrode_loc_zs = list(np.linspace(radii[0] - 100, radii[-1], num_electrodes))
    electrode_loc_zs.insert(1, radii[0])
    electrode_loc_zs.sort()
    num_electrodes += 1
    electrode_locs = np.zeros((num_electrodes, 3))

    electrode_locs[:, 2] = np.array(electrode_loc_zs)

    electrodeParams = {'sigma': 0.3,
                        'x': electrode_locs[:,0],
                        'y': electrode_locs[:,1],
                        'z': electrode_locs[:,2],
                        }

    num_cells = len(cell_dict['morphs'])

    for i in range(num_cells):
        cell_name = cell_dict['cellnames'][i]
        morphology = cell_dict['morphs'][i]
        cell_model = cell_dict['active_models'][i]
        print(cell_name)
        soma_pos = cell_dict['soma_positions'][i]
        syn_idcs = cell_dict['syn_idcs'][i]

        num_syns = len(syn_idcs)

        # set cell and synapse parameters
        cell_parameters, synapse_parameters = set_parameters(morphology, cell_model)
        [xrot, yrot, zrot] = cell_dict['rots'][i]
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
        Pz_traces = []
        # get data from num_syns simulations
        for j in range(num_syns):
            if not divmod(j, SIZE)[1] == RANK:
                continue
            print('syn number:', j)
            cell = create_cell(cell_parameters, active=active,
                               x_rot=xrot, y_rot=yrot, z_rot=zrot)

            ## if you know synidx:
            syn_idx = syn_idcs[j]
            # if you only know synapse location:
            # syn_loc = syn_idcs[j]
            # syn_idx = cell.get_closest_idx(x=syn_loc[0], y=syn_loc[1], z=syn_loc[2])

            cell, synapse, electrode_array = simulate(cell, [syn_idx], syn_input_time)
            print('cell simulated')
            cell.set_pos(x=soma_pos[0], y=soma_pos[1], z=soma_pos[2])
            syn_loc = (cell.xmid[syn_idx], cell.ymid[syn_idx], cell.zmid[syn_idx])

            synlocs.append((syn_loc[0], syn_loc[1], syn_loc[2]))

            # For active cells, we must subtract DC-dipole
            t_idx_before_input = np.argmin(np.abs(cell.tvec - syn_input_time)) - 1
            dipoles = cell.current_dipole_moment
            p_dc = dipoles[t_idx_before_input]
            dipoles -= p_dc

            # # P_traces.append(np.linalg.norm(dipoles, axis=1))
            Pz_traces.append(dipoles[:,2])
            # compute timepoint with biggest dipole
            timemax = [np.argmax(np.linalg.norm(np.abs(dipoles), axis=1))]
            t_max_list.append(timemax)
            p = dipoles[timemax]
            # compute LFP with single dipole
            # dip_loc = get_mass_center(cell, timemax)
            dip_loc = get_dipole_loc(soma_pos, syn_loc)
            lfp_single_dip = fs.calc_potential(p, dip_loc)
            print('pot from single dip computed')
            # compute LFP with multi-dipole

            # For active cells, we must subtract DC-dipole
            # multi_p_0, multi_p_locs = cell.get_multi_current_dipole_moments([0])
            multi_p_319, multi_p_locs = cell.get_multi_current_dipole_moments([t_idx_before_input])
            multi_p_dc = multi_p_319
            multi_p, multi_p_locs = cell.get_multi_current_dipole_moments(timemax)
            multi_p -= multi_p_dc

            Ni, Nt, Nd = multi_p.shape
            lfp_multi_dip = np.zeros((num_electrodes, Nt))
            for num in range(Ni):
                pot = fs.calc_potential(multi_p[num], multi_p_locs[num])
                lfp_multi_dip += pot


            # lfp_multi_dip = fs.calc_potential_from_multi_dipoles(cell, timemax)
            print('pot from multi dip computed')
            # compute relative errors
            RE = np.abs((lfp_single_dip - lfp_multi_dip)/lfp_multi_dip)

            print('syn_idx: ', syn_idx)
            print('syn_pos: ', syn_loc)
            print("RE EEG: ", RE[-1])
            p_list.append(p)
            p_loc_list.append(dip_loc)
            lfp_single_dip_list.append(lfp_single_dip)
            lfp_multi_dip_list.append(lfp_multi_dip)
            RE_list.append(RE)

            filename = join(savefolder, 'data_fig2_cellname:{}_active:{}_synidx:{}.npz'.format(cell_name, str(active), syn_idx))
            np.savez(filename,
                     lfp_multi = lfp_multi_dip,
                     lfp_single = lfp_single_dip,
                     dipole = p,
                     dip_loc = dip_loc,
                     sigmas = sigmas,
                     radii = radii,
                     syn_loc = syn_loc,
                     syn_idx = syn_idx,
                     zips = [list(zip(x, z)) for x, z in cell.get_idx_polygons()],
                     zmax = np.max(cell.zend),
                     soma_pos = soma_pos,
                     tvec = cell.tvec,
                     timemax = timemax,
                     electrode_locs = electrode_locs,
                     Pz_trace = dipoles[:, 2],
                     soma_vmem = cell.somav)


            # ### uncomment if you want to make fig1 for each synapse location
            # from fig_dipole_field import make_data, make_fig_1
            # dipole_results_dict = make_data(morphology, [syn_idx], dip_loc=dip_loc-cell.somapos, cell_model=cell_model, x_rot=xrot, y_rot=yrot, z_rot=zrot, active=active)
            # # cell1, cb_LFP_close, cb_LFP_far, multi_dip_LFP_close, multi_dip_LFP_far, db_LFP_close, db_LFP_far, LFP_max_close, LFP_max_far, time_max, multi_dips, multi_dip_locs, single_dip, r_mid, X, Z, X_f, Z_f = make_data(morphology, [syn_idx], dip_loc=dip_loc-cell.somapos, cell_model=cell_model, x_rot=xrot, y_rot=yrot, z_rot=zrot, active=False)
            # fig = make_fig_1(**dipole_results_dict)
            # fig1_title = './figures/test_figs/fig_dipole_field_{}.png'.format(syn_idx)
            # # fig1_title = './figures/test_figs/fig_dipole_field_segev_passive' + str(syn_idx) + '.png'
            # fig.savefig(fig1_title, bbox_inches='tight', dpi=300, transparent=True)
            #
            # # # uncomment if you want to check imems
            # plt.close('all')
            # fig2 = plt.figure()
            # ax1 = plt.subplot2grid((1,3),(0,0))
            # ax2 = plt.subplot2grid((1,3),(0,1))
            # ax3 = plt.subplot2grid((1,3),(0,2))
            #
            # ind_colors = plt.cm.viridis(np.linspace(0,1.,num=cell1.totnsegs))
            # zips = []
            # for x, z in cell1.get_idx_polygons():
            #     zips.append(list(zip(x, z)))
            # polycol = PolyCollection(list(zips), edgecolors = 'none', facecolors = ind_colors)
            # ax1.add_collection(polycol)
            # ax1.plot(cell1.xmid[syn_idx], cell1.zmid[syn_idx], 'ro')
            # ax1.set_title('morphology')
            #
            # [ax2.plot(cell1.tvec, cell1.imem[idx,:], color=ind_colors[idx], lw=0.5) for idx in range(cell1.totnsegs)]
            # ax2.set_xlabel('t (ms)')
            # ax2.set_ylabel('I (nA)')
            # ax2.set_title('imem')
            #
            # [ax3.plot(cell1.tvec, cell1.imem[idx,:], color=ind_colors[idx], lw=0.2) for idx in reversed(range(cell1.totnsegs))]
            # ax3.set_xlabel('t (ms)')
            # ax3.set_ylabel('I (nA)')
            # ax3.set_xlim([17, 20])
            # ax3.set_ylim([-0.0006, 0.0])
            # ax3.set_title('imem before input, zoomed')
            #
            # fig2.tight_layout(pad=.3, h_pad=.3, w_pad=-2)
            # fig2.set_size_inches(12,6)
            # # fig2.savefig('./figures/test_figs/fig_dipole_field_segev_active' + str(syn_idx) + 'imems.png', dpi=600)
            # fig2.savefig('./figures/test_figs/fig_dipole_field_segev_active' + str(syn_idx) + 'imems_zoomed.png', dpi=600)

        # k_100 = 100 # convert to percentage
        # eeg_ind = -1
        # ecog_ind = 2
        # RE_EEG = np.array(RE_list).reshape(num_syns, num_electrodes)[:,eeg_ind]*k_100
        # RE_ECoG = np.array(RE_list).reshape(num_syns, num_electrodes)[:,ecog_ind]*k_100
        #
        # zips = [zips.append(list(zip(x, z))) for x, z in cell.get_idx_polygons()]
        #
        # zmax = np.max(cell.zend)
        # soma_vmem = cell.vmem[0]
        #
        # filename = './data/data_fig2_' + cell_dict['cellnames'][i] + '_active_' + str(active)
        # np.savez(filename,
        #          lfp_multi = lfp_multi_dip_list,
        #          lfp_single = lfp_single_dip_list,
        #          re_eeg = RE_EEG,
        #          re_ecog = RE_ECoG,
        #          re = RE_list,
        #          dipoles = p_list,
        #          dip_locs = p_loc_list,
        #          sigmas = sigmas,
        #          radii = radii,
        #          synlocs = synlocs,
        #          syn_idxs = syn_idxs,
        #          zips = zips,
        #          zmax = zmax,
        #          rz = soma_pos,
        #          tvec = cell.tvec,
        #          t_max_list = t_max_list,
        #          electrode_locs = electrode_locs,
        #          Pz_traces = Pz_traces,
        #          soma_vmem = soma_vmem)



        # show synapse locations
        # zips_yz = []
        # for y, z in cell.get_idx_polygons(projection=('y', 'z')):
        #     zips_yz.append(list(zip(y, z)))

        # # plot morph with synlocs and vmem in soma
        # tvec = cell.tvec
        # plt.close('all')
        # fig = plt.figure()
        # ax_morph_xz = plt.subplot2grid((1,3),(0,0))
        # ax_morph_yz = plt.subplot2grid((1,3),(0,1))
        # ax_v_t = plt.subplot2grid((1,3),(0,2))
        #
        # clrs = plt.cm.viridis(np.linspace(0,0.8,num=num_syns))
        #
        # polycol = PolyCollection(list(zips), edgecolors = 'none', facecolors = 'k')
        # ax_morph_xz.add_collection(polycol)
        # polycol = PolyCollection(list(zips_yz), edgecolors = 'none', facecolors = 'k')
        # ax_morph_yz.add_collection(polycol)
        #
        # for sl in range(len(synlocs)):
        #     loc = synlocs[sl]
        #     ax_morph_xz.plot(loc[0], loc[2], 'o', ms = 3, c = clrs[sl])
        #     ax_morph_yz.plot(loc[1], loc[2], 'o', ms = 3, c = clrs[sl])
        # plt.axis('tight')
        # for ax in [ax_morph_xz, ax_morph_yz]:
        #     ax.axis('off')
        #     ax.set_aspect('equal')
        #
        # ax_v_t.plot(tvec, cell.vmem[0])
        # ax_v_t.set_xlim([tvec[0], tvec[-1]])
        # ax_v_t.set_ylim([-100, 60])
        # ax_v_t.set_xlabel('t (ms)')
        # ax_v_t.set_ylabel('V(t) in soma (mV)')
        #
        # title = 'weight = 0.002'
        # fig.text(0.2, 0.9, title)
        #
        # filename = './figures/test_figs/fig_test_synlocs_segev2018_spike' + cell_dict['cellnames'][i] + '.png'
        # print(filename)
        # plt.savefig(filename, dpi=300)
