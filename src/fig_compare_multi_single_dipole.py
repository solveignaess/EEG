import numpy as np
import matplotlib.pyplot as plt
import plotting_convention as plotting_convention
import LFPy
import matplotlib
from matplotlib import colorbar
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def set_parameters():
    """set cell, synapse and electrode parameters"""
    cell_parameters = {'morphology': './cell_models/segev/CNG_version/2013_03_06_cell03_789_H41_03.CNG.swc', #'./patdemo/cells/j4a.hoc',# only mandatory parameter
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

def simulate(celltype='l23'):
    """set synapse location. simulate cell, synapse and electrodes for input synapse location"""

    # create cell with parameters in dictionary
    cell = LFPy.Cell(**cell_parameters)
    if celltype == 'l23':
        cell.set_rotation(x=np.pi/2)

    pos = syn_loc
   # set synapse location
    synapse_parameters['idx'] = cell.get_closest_idx(x=pos[0], y=pos[1], z=pos[2])
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

def plot_neuron(axis, syn=False, lengthbar=False):
    zips = []
    for x, z in cell.get_idx_polygons():
        zips.append(zip(x, z))

    # faster way to plot points:
    polycol = PolyCollection(zips, edgecolors = 'none', facecolors = 'k')
    axis.add_collection(polycol)

    # small length reference bar
    if lengthbar:
        axis.plot([-500, -500], [-100, 900], 'k', lw=2, clip_on=False)
        axis.text(-430, 400, r'$1 \mathsf{mm}$', size = 8, va='center', ha='center', rotation = 'vertical')
    # axis.plot([100, 200], [-400, -400], 'k', lw=1, clip_on=False)
    # axis.text(150, -470, r'100$\mu$m', va='center', ha='center')

    plt.axis('tight')
    axis.axis('off')


if __name__ == '__main__':
    # make 4S-parameters
    sigmas = [0.3, 1.5, 0.015, 0.3]  #
    radii = [79000., 80000., 85000., 90000.]
    # set soma position
    rz = np.array([0., 0., 77500.])
    # make array of synapse positions
    num_syns = 2
    max_ind = 216
    syn_locs = [(0., 0., z) for z in np.linspace(-150., 750., num_syns)]
    # make electrode array params
    num_electrodes = 40
    electrode_locs = np.zeros((num_electrodes, 3))
    electrode_locs[:,2] = np.linspace(78600., radii[-1], num_electrodes) #[4000, 5150, 5450, 60000] #
    electrodeParams = {'sigma': 0.3,
                        'x': electrode_locs[:,0],
                        'y': electrode_locs[:,1],
                        'z': electrode_locs[:,2],
                        }

    # set cell and synapse parameters
    celltype = 'l23'
    cell_parameters, synapse_parameters = set_parameters()
    # create four-sphere class instance
    fs = LFPy.FourSphereVolumeConductor(radii, sigmas, electrode_locs)
    # lists for storing data:
    p_list = []
    lfp_single_dip_list = []
    lfp_multi_dip_list = []
    RE_list = []
    # get data from num_syns simulations
    for i in range(num_syns):
        syn_loc = syn_locs[i]
        cell, synapse, electrode_array = simulate()
        cell.set_pos(x=rz[0], y=rz[1], z=rz[2])

        # compute timepoint with biggest dipole
        dipoles = cell.current_dipole_moment
        timemax = [np.argmax(np.linalg.norm(np.abs(dipoles),axis=1))]
        p = dipoles[timemax]
        # compute LFP with single dipole
        lfp_single_dip = fs.calc_potential(p, rz)
        # compute LFP with multi-dipole
        lfp_multi_dip = fs.calc_potential_from_multi_dipoles(cell, timemax)
        # compute relative errors
        RE = np.abs((lfp_single_dip - lfp_multi_dip)/lfp_multi_dip)

        p_list.append(p)
        lfp_single_dip_list.append(lfp_single_dip)
        lfp_multi_dip_list.append(lfp_multi_dip)
        RE_list.append(RE)


    ################################################################################
    ######################################plot######################################
    ################################################################################
    plt.close('all')
    fig = plt.figure()
    # line colors
    clrs = plt.cm.viridis(np.linspace(0,1,num=num_syns))
    # head color
    head_colors = plt.cm.Pastel1([0,1,2,3])


    ax0 = plt.subplot2grid((2,4),(0,0), rowspan=2)
    ax1 = plt.subplot2grid((2,4),(0,2), colspan=2)
    ax2 = plt.subplot2grid((2,4),(1,2), colspan=2)

    # plot 4s-model
    for i in range(4):
        ax0.add_patch(plt.Circle((0, 0), radius = radii[i], color = head_colors[i], fill=False, lw = 1.))
        ax1.axvline(radii[i], color = head_colors[i])
        ax2.axvline(radii[i], color = head_colors[i])


    # ax0.annotate("brain",
    #             xy=(18*1e3, -50*1e3), xycoords='data',
    #             xytext=(100*1e3, -50*1e3), textcoords='data',
    #             size=10, va="center", ha="center",
    #             arrowprops=dict(arrowstyle="->",
    #                             connectionstyle="arc3,rad=-0.2",
    #                             fc="k"),
    #             )
    # ax0.annotate("csf",
    #             xy=(49.*1e3, -35*1e3), xycoords='data',
    #             xytext=(102*1e3, -35*1e3), textcoords='data',
    #             size=10, va="center", ha="center",
    #             arrowprops=dict(arrowstyle="->",
    #                             connectionstyle="arc3,rad=-0.2",
    #                             fc="k"),
    #             )
    # ax0.annotate("skull",
    #             xy=(66.*1e3, -20*1e3), xycoords='data',
    #             xytext=(114*1e3, -20*1e3), textcoords='data',
    #             size=10, va="center", ha="center",
    #             arrowprops=dict(arrowstyle="->",
    #                             connectionstyle="arc3,rad=-0.2",
    #                             fc="k"),
    #             )
    # ax0.annotate("scalp",
    #             xy=(80.*1e3, -5*1e3), xycoords='data',
    #             xytext=(122*1e3, -5*1e3), textcoords='data',
    #             size=10, va="center", ha="center",
    #             arrowprops=dict(arrowstyle="->",
    #                             connectionstyle="arc3,rad=-0.2",
    #                             fc="k"),
    #             )

    # plot morphology with synapses
    neuron_offset = 57000.
    plot_neuron(ax0)
    ax0.plot(0,0,'o', ms = 1e-4)
    # zoom in on neuron:
    zoom_ax = zoomed_inset_axes(ax0, 150, loc=5, bbox_to_anchor=(1100, 620)) # zoom = 6
    x1, x2, y1, y2 = -1000, 1000, 56000, 59200
    zoom_ax.set_xlim(x1, x2)
    zoom_ax.set_ylim(y1, y2)
    plot_neuron(zoom_ax, syn=True)
    mark_inset(ax0, zoom_ax, loc1=2, loc2=3, fc="None", ec=".5", lw=.4)
    # [i.set_linewidth(100) for i in zoom_ax.spines.itervalues()]
    zoom_ax.annotate('', xytext=(1000, 1200+neuron_offset),
                xycoords='data',
                xy=(0, 50+neuron_offset),
                arrowprops=dict(arrowstyle='wedge',
                                fc='gray',
                                # ec='gray'
                                lw = .2
                                )
                )
    zoom_ax.xaxis.set_ticks_position('none')
    zoom_ax.xaxis.set_ticklabels([])
    zoom_ax.yaxis.set_ticks_position('none')
    zoom_ax.yaxis.set_ticklabels([])
    for axis in ['top', 'left', 'bottom', 'right']:
        zoom_ax.spines[axis].set_color('r')
        zoom_ax.spines[axis].set_linewidth(200)


    ax0.axis('off')
    ax0.set_aspect('equal')

    # plot LFP-signals
    electrode_locs_z = electrode_locs[:,2]

    for i in range(num_syns):
        # plot lfps
        lfp_single_dip = lfp_single_dip_list[i].reshape(electrode_locs_z.shape)
        lfp_multi_dip = lfp_multi_dip_list[i].reshape(electrode_locs_z.shape)
        lfp_single_dip_log = np.sign(lfp_single_dip)*np.log10(np.abs(lfp_single_dip))
        lfp_multi_dip_log = np.sign(lfp_multi_dip)*np.log10(np.abs(lfp_multi_dip))
        # ax1.plot(electrode_locs_z, lfp_single_dip_log, color=clrs[i], label='single dipole')
        # ax1.plot(electrode_locs_z, lfp_multi_dip_log, '--', color=clrs[i], label='multi-dipole')
        ax1.plot(electrode_locs_z, lfp_single_dip, color=clrs[i], label='single dipole')
        ax1.plot(electrode_locs_z, lfp_multi_dip, '--', color=clrs[i], label='multi-dipole')


        # plot relative errors
        RE = RE_list[i].reshape(electrode_locs_z.shape)
        ax2.semilogy(electrode_locs_z, RE, color = clrs[i], label='RE')
        # show synapse locations
        zoom_ax.plot(syn_locs[i][0], syn_locs[i][2]+77500., 'o', color=clrs[i], ms = 5)

    ax1.legend(fontsize='xx-small')
    for ax in [ax1, ax2]:
        # fix axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim([np.min(electrode_locs_z), np.max(electrode_locs_z)])
    ax2.set_ylim([10**-4, 1])
    ax1.set_ylabel(r'$\Phi$ (mV)')
    ax2.set_ylabel('RE')
    fig.set_size_inches(10,4)
    plt.savefig('./figures/fig_compare_multi_single_dipole.png', dpi=300)
