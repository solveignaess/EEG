import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import plotting_convention as plotting_convention
import LFPy
import neuron
from matplotlib import colorbar
from matplotlib.collections import PolyCollection
import os
from os.path import join


def make_data(morphology, syninds, dip_loc=None, cell_model=None, x_rot=0,
              y_rot=0, z_rot=0, active=False):

    sigma = 0.3 #extracellular conductivity
    k = 1e6 # convert from mV to nV

    # compute LFP close to neuron
    # set up electrodes close to neuron
    X, Z = np.meshgrid(np.linspace(-550,550,101), np.linspace(-250,850,101))
    Y = np.zeros(X.shape)
    # simulate cell
    (cell_parameters, synapse_parameters,
     grid_electrode_parameters) = set_parameters(morphology, X, Y, Z,
                                         sigma=sigma,cell_model=cell_model)
    cell, synapse, grid_electrode_LFP = simulate(cell_parameters,
                                             synapse_parameters,
                                             grid_electrode_parameters,
                                             syninds, x_rot=x_rot, y_rot=y_rot,
                                             z_rot=z_rot, active=active)

    # multicompartment modeling of LFP close to cell
    cb_LFP_close = grid_electrode_LFP*k

    # multi-dipole modeling of LFP close to cell
    multi_dips, multi_dip_locs = cell.get_multi_current_dipole_moments()
    inf_vol = LFPy.InfiniteVolumeConductor(sigma)
    gridpoints_close = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    multi_dip_LFP_close = inf_vol.get_multi_dipole_potential(cell,
                                                         gridpoints_close)*k

    # single-dipole modeling of LFP close to cell
    cdm = LFPy.CurrentDipoleMoment(cell)
    single_dip = cdm.get_transformation_matrix() @ cell.imem
    syninds = cell.synidx
    if dip_loc is not None:
        r_mid = dip_loc
    else:
        r_soma_syns = [cell.get_intersegment_vector(idx0 = 0,
                       idx1 = i) for i in syninds]
        r_mid = np.average(r_soma_syns, axis = 0)
        r_mid = r_mid/2. + cell.somapos

    db_LFP_close = inf_vol.get_dipole_potential(single_dip ,
                                                gridpoints_close - r_mid)*k

    # compute LFP far from neuron
    # set up electrodes
    X_f,Z_f = np.meshgrid(np.linspace(-15000, 15001, 101),
                          np.linspace(-15000, 15000, 101))
    Y_f = np.zeros(X.shape)
    grid_electrode_parameters = {'x': X_f.flatten(),
                                 'y': Y_f.flatten(),
                                 'z': Z_f.flatten()
                                 }
    # simulate cell
    cell, synapse, grid_electrode_far_LFP = simulate(cell_parameters,
                                                 synapse_parameters,
                                                 grid_electrode_parameters,
                                                 syninds,
                                                 x_rot=x_rot,
                                                 y_rot=y_rot,
                                                 z_rot=z_rot, active=active)

    # multicompartment modeling of LFP far from cell
    cb_LFP_far = grid_electrode_far_LFP*k
    # multi-dipole modeling of LFP far from cell
    gridpoints_far = np.array([X_f.flatten(), Y_f.flatten(), Z_f.flatten()]).T
    multi_dip_LFP_far = inf_vol.get_multi_dipole_potential(cell, gridpoints_far)*k
    # single-dipole modeling of LFP far from cell
    db_LFP_far = inf_vol.get_dipole_potential(single_dip , gridpoints_far-r_mid)*k

    cdm = LFPy.CurrentDipoleMoment(cell)
    single_dip = cdm.get_transformation_matrix() @ cell.imem
    time_max = np.argmax(np.abs(np.linalg.norm(single_dip, axis=0)))
    LFP_max = 100. #nV #np.round(np.max(np.abs(grid_electrode_LFP[:, time_max])))

    return (cell, cb_LFP_close, cb_LFP_far, multi_dip_LFP_close,
            multi_dip_LFP_far, db_LFP_close, db_LFP_far, LFP_max, time_max,
            multi_dips, multi_dip_locs, single_dip, r_mid, X, Z, X_f, Z_f)


def set_parameters(morphology, X, Y, Z, sigma=0.3, cell_model=None):
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
                      'weight': 0.002, # 0.001, # synapse weight
                      'record_current': True, # record synapse current
                      # parameters not included in first version of fig
                      'syntype': 'Exp2Syn',
                      'tau1': 1., #Time constant, rise
                      'tau2': 3., #Time constant, decay
                      }

    grid_electrode_parameters = {'sigma': sigma,
                                 'x': X.flatten(),
                                 'y': Y.flatten(),
                                 'z': Z.flatten()
                                 }
    return cell_parameters, synapse_parameters, grid_electrode_parameters

def simulate(cell_parameters, synapse_parameters, grid_electrode_parameters,
             syninds, x_rot=0, y_rot=0, z_rot=0, active=False):
    """set synapse location. simulate cell, synapse and electrodes for
    input synapse location"""

    # create cell with parameters in dictionary
    if not active:
        cell = LFPy.Cell(**cell_parameters)
    else:
        cell = LFPy.TemplateCell(**cell_parameters)

    # rotate cell
    cell.set_rotation(x=x_rot)
    cell.set_rotation(y=y_rot)
    cell.set_rotation(z=z_rot)

    # insert synapses
    if type(syninds) == int:
        syninds = [syninds]
    for idx in syninds:
        synapse_parameters['idx'] = idx
        synapse = LFPy.Synapse(cell, **synapse_parameters)
        synapse.set_spike_times(np.array([20.]))

    # simulate cell
    cell.simulate(rec_imem=True, rec_vmem=True)

    #create grid electrodes
    grid_electrode = LFPy.RecExtElectrode(cell, **grid_electrode_parameters)
    grid_electrode_LFP = grid_electrode.get_transformation_matrix() @ cell.imem
    #grid_electrode.calc_lfp()

    return cell, synapse, grid_electrode_LFP

if __name__ == '__main__':
    # compute LFP close to and further away from l23 Eyal 2018 cell
    morphology = '2013_03_06_cell03_789_H41_03.ASC'
    # synaptic input index
    syninds = [481]
    # cell rotation
    [xrot, yrot, zrot] = [-np.pi/2, -np.pi/7, 0]
    # generate data
    (cell, cb_LFP_close, cb_LFP_far, multi_dip_LFP_close, multi_dip_LFP_far,
     db_LFP_close, db_LFP_far, LFP_max, time_max, multi_dips, multi_dip_locs,
     single_dip, r_mid, X, Z, X_far, Z_far) = make_data(morphology, syninds,
                                                        x_rot=xrot, y_rot=yrot)


    morph_zips = []
    for x, z in cell.get_idx_polygons():
        morph_zips.append(list(zip(x, z)))

    np.savez('./data/data_fig2',
             totnsegs=cell.totnsegs,
             imem=cell.imem,
             syninds=cell.synidx,
             xmids=cell.x.mean(axis=1),
             zmids=cell.z.mean(axis=1),
             morph_zips=morph_zips,
             cb_LFP_close=cb_LFP_close,
             cb_LFP_far=cb_LFP_far,
             multi_dip_LFP_close=multi_dip_LFP_close,
             multi_dip_LFP_far=multi_dip_LFP_far,
             db_LFP_close=db_LFP_close,
             db_LFP_far=db_LFP_far,
             LFP_max=LFP_max,
             time_max=time_max,
             multi_dips=multi_dips,
             multi_dip_locs=multi_dip_locs,
             single_dip=single_dip,
             r_mid=r_mid,
             X=X,
             Z=Z,
             X_far=X_far,
             Z_far=Z_far
             )
