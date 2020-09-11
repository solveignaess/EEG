import matplotlib
matplotlib.use("AGG")
import numpy as np
import matplotlib.pyplot as plt
import plotting_convention as plotting_convention
import LFPy
import neuron
from matplotlib import colorbar
from matplotlib.collections import PolyCollection
import os
from os.path import join


def make_data(morphology, syninds, dip_loc=None, cell_model=None, x_rot=0, y_rot=0, z_rot=0, active=False):
    l23 = True
    sigma = 0.3
    print('cell_model:', cell_model)
    # # compute LFP close to neuron
    # X,Z = np.meshgrid(np.linspace(-1300,1301,101), np.linspace(-900,1800,101))
    # Y = np.zeros(X.shape)
    # compute LFP very close to neuron
    X, Z = np.meshgrid(np.linspace(-550,550,101), np.linspace(-250,850,101))
    Y = np.zeros(X.shape)
    cell_parameters, synapse_parameters, grid_electrode_parameters = set_parameters(morphology, X, Y, Z, cell_model=cell_model)
    cell, synapse, grid_electrode = simulate(cell_parameters, synapse_parameters,
                                             grid_electrode_parameters, syninds,
                                             x_rot=x_rot, y_rot=y_rot, z_rot=z_rot,
                                             active=active)
    ## multicompartment
    cb_LFP_close = grid_electrode.LFP*1e6

    ## multi-dipole
    multi_dips, multi_dip_locs = cell.get_multi_current_dipole_moments()
    inf_vol = LFPy.InfiniteVolumeConductor(sigma)
    gridpoints_close = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    multi_dip_LFP_close = inf_vol.get_multi_dipole_potential(cell, gridpoints_close)*1E6
    ## save time when testing:
    # multi_dip_LFP_close = np.zeros(cb_LFP_close.shape)

    ## single dipole
    single_dip = cell.current_dipole_moment
    syninds = cell.synidx
    if dip_loc is not None:
        r_mid = dip_loc
    else:
        r_soma_syns = [cell.get_intersegment_vector(idx0 = 0,
                       idx1 = i) for i in syninds]
        r_mid = np.average(r_soma_syns, axis = 0)
        r_mid = r_mid/2. + cell.somapos

    db_LFP_close = inf_vol.get_dipole_potential(single_dip , gridpoints_close - r_mid)*1e6

    # compute LFP far from neuron
    X_f,Z_f = np.meshgrid(np.linspace(-15000,15001,101), np.linspace(-15000,15000,101))
    Y_f = np.zeros(X.shape)
    grid_electrode_parameters = {'x': X_f.flatten(),
                                 'y': Y_f.flatten(),
                                 'z': Z_f.flatten()
                                 }
    cell, synapse, grid_electrode_far = simulate(cell_parameters, synapse_parameters,
                                                 grid_electrode_parameters, syninds,
                                                 x_rot=x_rot, y_rot=y_rot, z_rot=z_rot,
                                                 active=active)
    ## multicompartment
    cb_LFP_far = grid_electrode_far.LFP*1e6
    ## multi dipole
    gridpoints_far = np.array([X_f.flatten(), Y_f.flatten(),Z_f.flatten()]).T
    multi_dip_LFP_far = inf_vol.get_multi_dipole_potential(cell, gridpoints_far)*1e6
    # save time when testing:
    # multi_dip_LFP_far = np.zeros(cb_LFP_far.shape)

    ## single dipole
    db_LFP_far = inf_vol.get_dipole_potential(single_dip , gridpoints_far-r_mid)*1e6

    max_ind = 10262844  # np.argmax(np.abs(grid_electrode_LFP)) + 100
    time_max = np.argmax(np.abs(np.linalg.norm(cell.current_dipole_moment, axis=1)))  # 334  # np.mod(max_ind, len(cell.tvec))
    LFP_max_close = 100.  #np.round(np.max(np.abs(grid_electrode_LFP[:, time_max])))
    LFP_max_far = 100.
    print('-'*200)

    results_dict = dict(cell=cell,
                        cb_LFP_close=cb_LFP_close,
                        cb_LFP_far=cb_LFP_far,
                        multi_dip_LFP_close=multi_dip_LFP_close,
                        multi_dip_LFP_far=multi_dip_LFP_far,
                        db_LFP_close=db_LFP_close,
                        db_LFP_far=db_LFP_far,
                        LFP_max_close=LFP_max_close,
                        LFP_max_far=LFP_max_far,
                        time_max=time_max,
                        multi_dips=multi_dips,
                        multi_dip_locs=multi_dip_locs,
                        single_dip=single_dip,
                        r_mid=r_mid,
                        X=X,
                        Z=Z,
                        X_f=X_f,
                        Z_f=Z_f)

    return results_dict

def set_parameters(morphology, X, Y, Z, cell_model=None):
    """set cell, synapse and electrode parameters"""
    # cell_parameters = {'morphology': morphology,
    #                'tstart': -10., # simulation start time
    #                'tstop': 100, # simulation stop time [ms]
    #                # parameters not included in first version of fig
    #                'dt': 2**-4,
    #                'passive': True,
    #                'passive_parameters': {'g_pas' : 1./21400, 'e_pas' : -68.851}, # S/cm^2, mV # new
    #                'Ra': 282, # Ω cm
    #                'cm': 0.49 # µF/cm^2
    #                }

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

        # if celltype == 'l23':
        #     cell_parameters['passive_parameters'] = {'g_pas' : 1./21400, 'e_pas' : -68.851} # S/cm^2, mV
        #     cell_parameters['Ra'] = 282 # Ω cm
        #     cell_parameters['cm'] = 0.49 # µF/cm^2
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

    #Guessing the electrode requires a 3D-grid. Cannot set 3D-grid in mgrid,
    # because we need X, Z and LFP to be 2D when plotting.(?)
    grid_electrode_parameters = {'sigma': 0.3,
                                 'x': X.flatten(),
                                 'y': Y.flatten(),
                                 'z': Z.flatten()
                                 }
    return cell_parameters, synapse_parameters, grid_electrode_parameters

def simulate(cell_parameters, synapse_parameters, grid_electrode_parameters, syninds, x_rot=0, y_rot=0, z_rot=0, active=False):
    """set synapse location. simulate cell, synapse and electrodes for input synapse location"""

    # create cell with parameters in dictionary
    if not active:
        cell = LFPy.Cell(**cell_parameters)
    else:
        cell = LFPy.TemplateCell(**cell_parameters)

    cell.set_rotation(x=x_rot)
    cell.set_rotation(y=y_rot)
    cell.set_rotation(z=z_rot)

    if type(syninds) == int:
        syninds = [syninds]
    for idx in syninds:
        print('idx', idx)
        synapse_parameters['idx'] = idx
        print("synapse_parameters['idx']", synapse_parameters['idx'])
        # synapse_parameters['idx'] = 551 #cell.get_closest_idx(x=pos[0], y=pos[1], z=pos[2])
        # create synapse with parameters in dictionary
        synapse = LFPy.Synapse(cell, **synapse_parameters)
        synapse.set_spike_times(np.array([20.]))
    #  simulation goes from t: 0-100 in ms. spike_time = 20ms
    # timeres = 0.1 --> 801 measurements!

    cell.simulate(rec_imem = True,
                  rec_vmem = True,
                  rec_current_dipole_moment=True)

    #create grid electrodes
    grid_electrode = LFPy.RecExtElectrode(cell, **grid_electrode_parameters)
    grid_electrode.calc_lfp()

    return cell, synapse, grid_electrode

def make_fig_1(cell,
               cb_LFP_close, cb_LFP_far,
               multi_dip_LFP_close, multi_dip_LFP_far,
               db_LFP_close, db_LFP_far,
               LFP_max_close, LFP_max_far,
               time_max,
               multi_dips, multi_dip_locs,
               single_dip, r_mid,
               X, Z, X_f, Z_f):
    plt.interactive(1)
    plt.close('all')
    colorbrewer = {'lightblue': '#a6cee3', 'blue': '#1f78b4', 'lightgreen': '#b2df8a',
                   'green': '#33a02c', 'pink': '#fb9a99', 'red': '#e31a1c',
                   'lightorange': '#fdbf6f', 'orange': '#ff7f00',
                   'lightpurple': '#cab2d6', 'purple': '#6a3d9a',
                   'yellow': '#ffff33', 'brown': '#b15928'}

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

    # ax0.set_title('transmembrane currents')
    ax0.set_ylabel('neuron simulation', labelpad=-.2)
    ax1.set_ylabel(r'$\Phi$' + ' close-up')
    ax2.set_ylabel(r'$\Phi$' + ' zoomed out')

    # plot neuron morphology in top row
    plot_neuron(ax0, cell, syn=True, lengthbar=True)
    plot_neuron(ax3, cell, syn=True, lengthbar=True, lb_clr='w')
    plot_neuron(ax6, cell, syn=True, lengthbar=True, lb_clr='w')

    # plot transmembrane currents
    for idx in range(cell.totnsegs):
        arrowlength = np.abs(cell.imem[idx, time_max])*1e5
        # print idx, arrowlength
        wdth = 1.
        if [idx] == cell.synidx:
            print(idx, arrowlength)
            arrowlength = -700.
            wdth = 2.
            ax0.arrow(cell.xmid[idx]-arrowlength, cell.zmid[idx],
                       arrowlength, 0.,
                       width = 4.,
                       head_length = 39.,
                       head_width = 30.,
                       length_includes_head = True, color='#0D325F',
                    #    alpha=.5
                       )
        else:
            ax0.arrow(cell.xmid[idx], cell.zmid[idx],
                       arrowlength, 0.,
                       width = wdth,
                       head_length = 3.4,
                       head_width = 7.,
                       length_includes_head = True, color='#D90011',
                       alpha=.5)

    # plt lfp close
    plot_lfp(fig, ax1, cb_LFP_close, LFP_max_close, time_max, X, Z, lengthbar=True)
    plot_lfp(fig, ax4, multi_dip_LFP_close, LFP_max_close, time_max, X, Z, lengthbar=False)
    plot_lfp(fig, ax7, db_LFP_close, LFP_max_close, time_max, X, Z, lengthbar=False)

    # plot lfp far
    plot_lfp_far(fig, ax2, cb_LFP_far, LFP_max_far, time_max, X_f, Z_f, lengthbar=True)
    plot_lfp_far(fig, ax5, multi_dip_LFP_far, LFP_max_far, time_max, X_f, Z_f, lengthbar=False)
    LFP, levels, ep_intervals, ticks = plot_lfp_far(fig, ax8, db_LFP_far, LFP_max_far, time_max, X_f, Z_f, lengthbar=False, colorax=True)

    # plot neurons in second row
    for ax in [ax1, ax4]:
        plot_neuron(ax, cell, syn=False, clr='w')

    # plot multi dipole arrows
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
    # if l23:
    arrow = single_dip[time_max]*25  # np.sum(P, axis = 0)*0.12
    # else:
        # arrow = single_dip[time_max]*50  # np.sum(P, axis = 0)*0.12
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
    # cax.plot(np.arange(100), np.arange(100))
    cbar = fig.colorbar(ep_intervals,cax=cax,
                orientation='horizontal', format='%3.3f',
                extend = 'max')
                                # extend = 'max')
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
    # plt.tight_layout()

    # plotting_convention.simplify_axes(fig.axes)
    plotting_convention.mark_subplots([ax0, ax3, ax6], letters='ABC', xpos=-0.02, ypos=1.0)
    plotting_convention.mark_subplots([ax1, ax4, ax7], letters='DEF', xpos=-0.02, ypos=1.05)
    plotting_convention.mark_subplots([ax2, ax5, ax8], letters='GHI', xpos=-0.02, ypos=0.94)
    for ax in [ax1, ax2, ax4, ax5, ax7, ax8, ax0, ax3, ax6]:
        ax.set_aspect('equal', 'datalim')
    plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.1, bottom=0.05, right=0.96, top=0.93)
    return fig

def plot_lfp(fig, ax, LFP_measurements, max_LFP, timestep, X, Z, colorax = False,
             lengthbar = False):
    print('plot_lfp called')
    if LFP_measurements.size > X.size:
        print('timestep:', timestep)
        LFP_measurements = LFP_measurements[:,timestep]
    LFP = np.array(LFP_measurements).reshape(X.shape)
    print('LFP:', LFP)
    LFP_norm = LFP/max_LFP
    print('LFP_norm:', LFP_norm)
    # print LFP_norm
    # scalp levels:
    # num = 5
    num = 9
    levels = np.logspace(-4, 0, num = num)
    print('levels:', levels)
    # levels = np.linspace(0.1,1,num)
    levels_norm = np.concatenate((-levels[::-1], levels))
    rainbow_cmap = plt.cm.get_cmap('PRGn') # rainbow, spectral, RdYlBu
    colors_from_map = [rainbow_cmap(i*np.int(255/(len(levels_norm) - 2))) for i in range(len(levels_norm) -1)]
    colors_from_map[num - 1] = (1.0, 1.0, 1.0, 1.0)
    ticks = [levels_norm[2*i] for i in range(int(num/2 + 1))] + [levels_norm[int(num + 2*i)] for i in range(int(num/2 + 1))]
    ep_intervals = ax.contourf(X, Z, LFP_norm,# vmin=-200, vmax=200,
                               zorder=-2, colors = colors_from_map,
                               levels=levels_norm, extend = 'both') #norm = LogNorm())#,
                                          # norm = SymLogNorm(1E-30))#, vmin = -40, vmax = 40))

    ax.contour(X, Z, LFP_norm, lw = 0.4,  # 20,
               colors='k', zorder = -2,  # extend='both')
               levels=levels_norm)

    if lengthbar:
        ax.plot([-400, -400], [-200, 800], 'k', lw=2, clip_on=False)
        ax.text(-330, 400, r'$1 \mathsf{mm}$', color='k', size = 8, va='center', ha='center', rotation = 'vertical')
        # ax.plot([-1200, -1200], [-800, 200], 'k', lw=2, clip_on=False)
        # ax.text(-1090, -380, r'$1 \mathsf{mm}$', size = 8, va='center', ha='center', rotation = 'vertical')
    # plt.axis('tight')
    ax.axis('off')
    ax.set_xlim([-500,500])
    ax.set_ylim([-250,850])
    return LFP, levels, ep_intervals

def plot_lfp_far(fig, ax, LFP_measurements, max_LFP, timestep, X_f, Z_f, colorax = False,
             lengthbar = False):
    # print 'lengthbar', lengthbar
    # print 'far input', LFP_measurements[:, timestep]
    if LFP_measurements.size > X_f.size:
        LFP_measurements = LFP_measurements[:,timestep]
    LFP = np.array(LFP_measurements).reshape(X_f.shape)
    # print LFP
    LFP_norm = LFP/max_LFP
    # print LFP_norm
    # scalp levels:
    num = 9
    # num = 5
    levels = np.logspace(-4, 0, num = num)
    # levels = np.linspace(0.1,1,num)
    levels_norm = np.concatenate((-levels[::-1], levels))
    rainbow_cmap = plt.cm.get_cmap('PRGn') # rainbow, spectral, RdYlBu
    colors_from_map = [rainbow_cmap(i*np.int(255/(len(levels_norm) - 2))) for i in range(len(levels_norm) -1)]
    colors_from_map[num - 1] = (1.0, 1.0, 1.0, 1.0)
    ticks = [levels_norm[2*i] for i in range(int(num/2 + 1))] + [levels_norm[int(num + 2*i)] for i in range(int(num/2 + 1))]

    ep_intervals = ax.contourf(X_f, Z_f, LFP_norm,# vmin=-200, vmax=200,
                               zorder=-2, colors = colors_from_map,
                               levels=levels_norm, extend = 'both') #norm = LogNorm())#,
                                          # norm = SymLogNorm(1E-30))#, vmin = -40, vmax = 40))
    ep_lines = ax.contour(X_f, Z_f, LFP_norm, lw = 0.4,  # 20,
               colors='k', zorder = -2,  # extend='both')
               levels=levels_norm)

    # plt.axis('tight')
    if lengthbar:
        print('lengthbar true')
        ax.plot([-14000, -14000], [-14000, -13000], 'k', lw=2, clip_on=False)
        ax.text(-11700, -13600, r'$1 \mathsf{mm}$', size = 8, va='center', ha='center')

    ax.axis('off')

    return LFP, levels, ep_intervals, ticks

def plot_neuron(axis, cell, syn=False, lengthbar=False, clr='k', lb_clr='k'):
    zips = []
    for x, z in cell.get_idx_polygons():
        zips.append(list(zip(x, z)))
    # faster way to plot points:
    polycol = PolyCollection(list(zips), edgecolors = 'none', facecolors = 'k')
    axis.add_collection(polycol)
    # small length reference bar
    if lengthbar:
        axis.plot([-400, -400], [-200, 800], lb_clr, lw=2, clip_on=False)
        axis.text(-330, 400, r'$1 \mathsf{mm}$', color=lb_clr, size = 8, va='center', ha='center', rotation = 'vertical')
    # axis.plot([100, 200], [-400, -400], 'k', lw=1, clip_on=False)
    # axis.text(150, -470, r'100$\mu$m', va='center', ha='center')

    # plt.axis('tight')
    axis.set_xlim([-500,500])
    axis.set_ylim([-250,850])
    axis.axis('off')

    # red dot where synapse is located, ms = markersize; number of points given as float:
    if syn:
        for idx_num, idx in enumerate(cell.synidx):
            axis.plot(cell.xmid[idx], cell.zmid[idx], 'o', ms=3,
                    markeredgecolor='k', markerfacecolor='r')

if __name__ == '__main__':

    # morphology = './cell_models/segev/CNG_version/2013_03_06_cell03_789_H41_03.CNG.swc'
    # morphology = './cell_models/segev/CNG_version/2013_03_06_cell03_789_H41_03.ASC'
    morphology = '2013_03_06_cell03_789_H41_03.ASC'
    # syn_loc = (-60, 0, 600)
    # syn_loc = (60, 0, 600)
    # syninds = [432]
    # syninds = [328]
    # syninds = [557]
    syninds = [481]
    [xrot, yrot, zrot] = [-np.pi/2, -np.pi/7, 0]
    dipole_fiel_data_dict = make_data(morphology, syninds, x_rot=xrot, y_rot=yrot)
    # print('time_max', time_max)
    fig = make_fig_1(**dipole_fiel_data_dict)
    # fig.savefig('./figures/fig_dipole_field.pdf', bbox_inches='tight', dpi=300, transparent=True)
    # fig.savefig('./figures/fig_dipole_field_passiveTrue_single_syn328.pdf', bbox_inches='tight', dpi=300, transparent=True)
    # fig.savefig('./figures/fig_dipole_field_passiveTrue_single_syn557.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('./figures/fig_dipole_field_passiveTrue_single_syn481_mod.pdf', bbox_inches='tight', dpi=300)
    # fig.savefig('./figures/fig_dipole_field_passiveFalse2.pdf', bbox_inches='tight', dpi=300, transparent=True)
