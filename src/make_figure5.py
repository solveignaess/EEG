import os
from os.path import join
import matplotlib
matplotlib.use("AGG")
import numpy as np
from plotting_convention import mark_subplots, simplify_axes
import matplotlib.pyplot as plt
import neuron
import LFPy
from neuron import h
np.random.seed(1234)


def run_cell_simulation(make_ca_spike, dt, cell_name, grid_elec_params,
                                            elec_params):

    T = 100

    if cell_name == 'almog':
        model_folder = join('cell_models', 'almog')

        neuron.load_mechanisms(model_folder)
        os.chdir(model_folder)
        cell_parameters = {
                'morphology': join('A140612.hoc'),
                'v_init': -62,
                'passive': False,
                'nsegs_method': None,
                'dt': dt,  # [ms] Should be a power of 2
                'tstart': -200,  # [ms] Simulation start time
                'tstop': T,  # [ms] Simulation end time
                'custom_code': [join('cell_model.hoc')] # Loads model specific code
        }

        cell = LFPy.Cell(**cell_parameters)
        os.chdir(join('..', '..'))
        cell.set_rotation(x=np.pi/2, y=0.1)
    elif cell_name == 'hay':

        model_folder = join('cell_models', 'hay', 'L5bPCmodelsEH')
        neuron.load_mechanisms(join(model_folder, "mod"))

        ##define cell parameters used as input to cell-class
        cellParameters = {
            'morphology': join(model_folder, "morphologies", "cell1.asc"),
            'templatefile': [join(model_folder, "models", "L5PCbiophys3.hoc"),
                             join(model_folder, "models", "L5PCtemplate.hoc")],
            'templatename': 'L5PCtemplate',
            'templateargs': join(model_folder, "morphologies", "cell1.asc"),
            'passive': False,
            'nsegs_method': None,
            'dt': dt,
            'tstart': -200,
            'tstop': T,
            'v_init': -60,
            'celsius': 34,
            'pt3d': True,
        }
        cell = LFPy.TemplateCell(**cellParameters)
        cell.set_rotation(x=np.pi/2, y=0.1)

    plot_idxs = [cell.somaidx[0], cell.get_closest_idx(z=500)]
    idx_clr = {idx: ['b', 'orange'][num] for num, idx in enumerate(plot_idxs)}

    if cell_name == 'hay':
        weights = [0.07, 0.15]
    elif cell_name == 'almog':
        weights = [0.05, 0.15]

    delay = 30
    synapse_s = LFPy.Synapse(cell, idx=cell.get_closest_idx(z=0),
                             syntype='Exp2Syn', weight=weights[0],
                             tau1=0.1, tau2=1.)
    synapse_s.set_spike_times(np.array([delay]))

    synapse_a = None
    if make_ca_spike:
        synapse_a = LFPy.Synapse(cell, idx=cell.get_closest_idx(z=400),
                                 syntype='Exp2Syn', weight=weights[1],
                                 tau1=0.1, tau2=10.)
        synapse_a.set_spike_times(np.array([delay]))
    cell.simulate(rec_imem=True, rec_vmem=True)
    print("MAX |I_mem(soma, apic)|: ", np.max(np.abs(cell.imem[plot_idxs]), axis=1))

    elec = LFPy.RecExtElectrode(cell, **elec_params)
    elec_LFP = 1e3 * elec.get_transformation_matrix() @ cell.imem
    elec_LFP -= elec_LFP[:, 0, None]

    grid_electrode = LFPy.RecExtElectrode(cell, **grid_elec_params)
    grid_LFP = 1e3 * grid_electrode.get_transformation_matrix() @ cell.imem
    grid_LFP -= grid_LFP[:, 0, None]

    cdm = LFPy.CurrentDipoleMoment(cell)
    cdm = cdm.get_transformation_matrix() @ cell.imem
    cdm -= cdm[:, 0, None]

    cell_data_dict = {"vmem": cell.vmem.copy(),
                      "tvec": cell.tvec.copy(),
                      "elec_LFP": elec_LFP.copy(),
                      "grid_LFP": grid_LFP.copy(),
                      "cdm": cdm.copy(),
                      "cell_x": cell.x.copy(),
                      "cell_z": cell.z.copy(),
    }
    synapse_s = None
    synapse_a = None
    cell.__del__()
    return cell_data_dict, idx_clr, plot_idxs


def animate_ca_spike():
    # Deprecated, and not updated for LFPy2.2 and newer
    dt = 2**-5
    cell_name = 'hay'

    jitter_std = 10
    num_trials = 1000

    # Create a grid of measurement locations, in (mum)
    grid_x, grid_z = np.mgrid[-750:751:25, -750:1301:25]
    grid_y = np.zeros(grid_x.shape)

    # Define electrode parameters
    grid_elec_params = {
        'sigma': 0.3,      # extracellular conductivity
        'x': grid_x.flatten(),  # electrode requires 1d vector of positions
        'y': grid_y.flatten(),
        'z': grid_z.flatten()
    }

    elec_x = np.array([30, ])
    elec_y = np.array([0, ])
    elec_z = np.array([0, ])

    num = 11
    levels = np.logspace(-2.5, 0, num=num)
    scale_max = 100.

    levels_norm = scale_max * np.concatenate((-levels[::-1], levels))
    bwr_cmap = plt.cm.get_cmap('bwr_r')  # rainbow, spectral, RdYlBu

    colors_from_map = [bwr_cmap(i * np.int(255 / (len(levels_norm) - 2)))
                       for i in range(len(levels_norm) - 1)]
    colors_from_map[num - 1] = (1.0, 1.0, 1.0, 1.0)

    cell, idx_clr, plot_idxs = run_cell_simulation(make_ca_spike=False,
                                                   dt=dt, cell_name=cell_name)
    grid_electrode = LFPy.RecExtElectrode(cell, **grid_elec_params)
    grid_electrode.calc_lfp()
    grid_LFP = 1e3 * grid_electrode.LFP

    grid_LFP -= grid_LFP[:, 0, None]
    for time_idx in range(len(cell.tvec))[2000:]:
        plt.close("all")
        fig = plt.figure(figsize=[5, 4])
        fig.text(0.5, 0.95, "T={:1.1f} ms".format(cell.tvec[time_idx]),
                 ha="center")
        ax_ = fig.add_axes([0.4, 0.14, 0.6, 0.88], xticks=[], yticks=[],
                           aspect=1, frameon=False)
        cax = fig.add_axes([0.5, 0.15, 0.4, 0.01])
        ax_vm = fig.add_axes([0.25, 0.65, 0.2, 0.3],
                             ylabel="membrane\npotential\n(mV)",
                             xlabel="time (ms)")
        ax_cdm = fig.add_axes([0.25, 0.2, 0.2, 0.3],
                              ylabel="current dipole\nmoment\n(µA$\cdot$µm)",
                              xlabel="time (ms)")
        grid_LFP_ = grid_LFP[:, time_idx].reshape(grid_x.shape)
        [ax_.plot([cell.xstart[idx], cell.xend[idx]],
        [cell.zstart[idx], cell.zend[idx]], c='gray')
            for idx in range(cell.totnsegs)]
        ax_.plot([350, 350], [-540, -640], c='k', lw=2)
        ax_.text(360, -590, "100 $\mu$m", va='center')
        ep_intervals = ax_.contourf(grid_x, grid_z, grid_LFP_,
                                   zorder=-2, colors=colors_from_map,
                                   levels=levels_norm, extend='both')

        ax_.contour(grid_x, grid_z, grid_LFP_, colors='k',
                    linewidths=(1), zorder=-2,
                   levels=levels_norm)

        cbar = fig.colorbar(ep_intervals, cax=cax, orientation='horizontal',
                            format='%.0E', extend='max')

        cbar.set_ticks(np.array([-1, -0.1, -0.01, 0, 0.01,  0.1, 1]) * scale_max)
        cax.set_xticklabels(np.array(np.array([-1, -0.1, -0.01, 0, 0.01, 0.1, 1]
                                              ) * scale_max, dtype=int),
                            fontsize=7, rotation=45)
        cbar.set_label('$\phi$ (µV)', labelpad=-5)
        [ax_vm.plot(cell.tvec, cell.vmem[idx], c=idx_clr[idx])
         for idx in plot_idxs]
        ax_cdm.plot(cell.tvec, cell.current_dipole_moment[:, 2], c='k')
        ax_vm.axvline(cell.tvec[time_idx], c='gray', lw=1, ls='--')
        ax_cdm.axvline(cell.tvec[time_idx], c='gray', lw=1, ls='--')
        simplify_axes([ax_vm, ax_cdm])
        anim_fig_folder = "anim_no_ca"
        os.makedirs(anim_fig_folder, exist_ok=True)
        plt.savefig(join(anim_fig_folder, "cell_ca_cont_{:04d}.png".format(
            time_idx)))


def simulate_spike_current_dipole_moment():

    dt = 2**-5
    cell_name = 'hay'

    jitter_std = 10
    num_trials = 1000

    # Create a grid of measurement locations, in (mum)
    grid_x, grid_z = np.mgrid[-750:751:25, -750:1301:25]

    grid_y = np.zeros(grid_x.shape)

    # Define electrode parameters
    grid_elec_params = {
        'sigma': 0.3,      # extracellular conductivity
        'x': grid_x.flatten(),  # electrode requires 1d vector of positions
        'y': grid_y.flatten(),
        'z': grid_z.flatten()
    }

    elec_x = np.array([30, ])
    elec_y = np.array([0, ])
    elec_z = np.array([0, ])

    elec_params = {
        'sigma': 0.3,  # extracellular conductivity
        'x': elec_x,  # x,y,z-coordinates of contact points
        'y': elec_y,
        'z': elec_z
    }

    elec_clr = 'r'

    cell_woca_data, idx_clr, plot_idxs = run_cell_simulation(
                                            make_ca_spike=False,
                                            dt=dt, cell_name=cell_name,
                                            grid_elec_params=grid_elec_params,
                                            elec_params=elec_params)
    cell_wca_data, idx_clr, plot_idxs = run_cell_simulation(
                                            make_ca_spike=True,
                                            dt=dt, cell_name=cell_name,
                                            grid_elec_params=grid_elec_params,
                                            elec_params=elec_params)

    fig = plt.figure(figsize=[8, 7])
    fig.subplots_adjust(hspace=0.5, left=0.0, wspace=0.4, right=0.96,
                        top=0.97, bottom=0.1)

    ax_m = fig.add_axes([-0.01, 0.05, 0.25, 0.97], aspect=1, frameon=False,
                        xticks=[], yticks=[])
    ax_m.plot(cell_wca_data["cell_x"].T, cell_wca_data["cell_z"].T, c='k')

    [ax_m.plot(cell_wca_data["cell_x"][idx].mean(),
               cell_wca_data["cell_z"][idx].mean(), 'o',
               c=idx_clr[idx], ms=13) for idx in plot_idxs]
    ax_m.plot(elec_x, elec_z, elec_clr, marker='D')

    ax_top = 0.98
    ax_h = 0.15
    h_space = 0.1
    ax_w = 0.17
    ax_left = 0.37

    num = 11
    levels = np.logspace(-2.5, 0, num=num)
    scale_max = 100.

    levels_norm = scale_max * np.concatenate((-levels[::-1], levels))
    bwr_cmap = plt.cm.get_cmap('bwr_r')

    colors_from_map = [bwr_cmap(i * np.int(255 / (len(levels_norm) - 2)))
                       for i in range(len(levels_norm) - 1)]
    colors_from_map[num - 1] = (1.0, 1.0, 1.0, 1.0)

    spike_plot_time_idxs = [1030, 1151]
    summed_cdm_max = np.zeros(2)
    for plot_row, cell_data in enumerate([cell_woca_data, cell_wca_data]):
        ax_left += plot_row * 0.25

        grid_LFP = cell_data["grid_LFP"]
        cdm = cell_data["cdm"]
        tvec = cell_data["tvec"]
        vmem = cell_data["vmem"]
        elec_LFP = cell_data["elec_LFP"]

        time_idx = spike_plot_time_idxs[plot_row]
        print(tvec[time_idx])

        grid_LFP_ = grid_LFP[:, time_idx].reshape(grid_x.shape)
        ax_ = fig.add_axes([0.75, 0.55 - plot_row * 0.46, 0.3, 0.45],
                           xticks=[], yticks=[], aspect=1, frameon=False)
        mark_subplots(ax_, [["D"], ["E"]][plot_row], ypos=0.95, xpos=0.35)

        ax_.plot(cell_data["cell_x"].T, cell_data["cell_z"].T, c='k')

        ep_intervals = ax_.contourf(grid_x, grid_z, grid_LFP_,
                                   zorder=-2, colors=colors_from_map,
                                   levels=levels_norm, extend='both')

        ax_.contour(grid_x, grid_z, grid_LFP_, colors='k', linewidths=(1),
                    zorder=-2, levels=levels_norm)

        if plot_row == 1:
            cax = fig.add_axes([0.82, 0.12, 0.16, 0.01])
            cbar = fig.colorbar(ep_intervals, cax=cax,
                                orientation='horizontal', format='%.0E')

            cbar.set_ticks(np.array([-1, -0.1, -0.01, 0, 0.01,  0.1, 1])
                           * scale_max)
            cax.set_xticklabels(np.array(np.array(
                [-1, -0.1, -0.01, 0, 0.01, 0.1, 1]) * scale_max, dtype=int),
                                fontsize=11, rotation=45)
            cbar.set_label('$\phi$ (µV)', labelpad=-5)

        sum_tvec, summed_cdm = sum_jittered_cdm(cdm[2, :],
                                            dt, jitter_std, num_trials)

        summed_cdm_max[plot_row] = np.max(np.abs(summed_cdm))

        ax_vm = fig.add_axes([ax_left, ax_top - ax_h, ax_w, ax_h],
                           ylim=[-80, 50], xlim=[0, 100])

        ax_eap = fig.add_axes([ax_left, ax_top - 2 * ax_h - h_space,
                               ax_w, ax_h],
                           ylim=[-120, 40], xlim=[0, 100])
        ax_cdm = fig.add_axes([ax_left, ax_top - 3 * ax_h - 2*h_space,
                               ax_w, ax_h],
                              ylim=[-0.5, 1], xlim=[0, 100],)

        ax_cdm_sum = fig.add_axes([ax_left, ax_top - 4 * ax_h - 3*h_space,
                                   ax_w, ax_h],
                              ylim=[-250, 100],
                              xlabel="Time (ms)", xlim=[0, 140])
        if plot_row == 0:
            ax_vm.set_ylabel("Membrane\npotential\n(mV)",
                             labelpad=-3)
            ax_eap.set_ylabel("Extracellular\npotential\n(µV)",
                              labelpad=-3)
            ax_cdm.set_ylabel("Curent dipole\nmoment\n(µA$\cdot$µm)",
                              labelpad=-3)
            ax_cdm_sum.set_ylabel("Jittered sum\n(µA$\cdot$µm)",
                                  labelpad=-3)
        elif plot_row == 1:
            ax_vm.text(65, -5, "Ca$^{2+}$ spike", fontsize=11, color='orange')
            ax_vm.arrow(80, -10, -10, -18, color='orange', head_width=4)

        mark_subplots(ax_vm, [["B1"], ["C1"]][plot_row],
                      xpos=-0.3, ypos=0.93)
        mark_subplots(ax_eap, [["B2"], ["C2"]][plot_row],
                      xpos=-0.3, ypos=0.93)
        mark_subplots(ax_cdm, [["B3"], ["C3"]][plot_row],
                      xpos=-0.35, ypos=0.97)
        mark_subplots(ax_cdm_sum, [["B4"], ["C4"]][plot_row],
                      xpos=-0.3, ypos=0.93)
        [ax_vm.plot(tvec, vmem[idx], c=idx_clr[idx]) for idx in plot_idxs]
        ax_vm.axvline(tvec[time_idx], ls='--', c='gray')

        [ax_eap.plot(tvec, elec_LFP[idx], c=elec_clr)
         for idx in range(len(elec_x))]

        ax_cdm.plot(tvec, 1e-3 * cdm[2, :], c='k')
        ax_cdm_sum.plot(sum_tvec, 1e-3 * summed_cdm, c='k')

    print("Summed CDM max (abs), ratio", summed_cdm_max * 1e-3, )

    mark_subplots([ax_m], xpos=0.1, ypos=0.95)
    simplify_axes(fig.axes)

    plt.savefig(join("figures", 'Figure5.pdf'))


def simulate_laminar_LFP():
    # DEPRECATED, and not updated for LFPy2.2 and newer
    dt = 2**-5
    cell_name = 'hay'

    elec_z = np.linspace(-200, 1200, 15)
    elec_x = np.ones(len(elec_z)) * 50
    elec_y = np.zeros(len(elec_z))

    h = np.abs(elec_z[1] - elec_z[0])

    electrode_parameters = {
        'sigma': 0.3,  # extracellular conductivity
        'x': elec_x,  # x,y,z-coordinates of contact points
        'y': elec_y,
        'z': elec_z
    }

    elec_clr = lambda idx: plt.cm.viridis(idx / len(elec_z))

    num_sims = 10

    cells = []

    summed_LFP = []
    summed_cdm = []

    for sim in range(num_sims):
        print(sim + 1, "/", num_sims)
        cell_wca, idx_clr, plot_idxs = run_cell_simulation_distributed_input(
            dt=dt, cell_name=cell_name)
        cells.append([cell_wca, idx_clr, plot_idxs])
    for sim in range(num_sims):
        cell_wca, idx_clr, plot_idxs = cells[sim]
        fig = plt.figure(figsize=[7, 7])
        fig.subplots_adjust(hspace=0.5, left=0.0, wspace=0.4, right=0.96,
                            top=0.97, bottom=0.1)

        ax_m = fig.add_axes([-0.01, 0.05, 0.3, 0.97], aspect=1, frameon=False,
                            xlim=[-350, 350],
                            xticks=[], yticks=[])
        [ax_m.plot([cell_wca.xstart[idx], cell_wca.xend[idx]],
                  [cell_wca.zstart[idx], cell_wca.zend[idx]], c='k')
         for idx in range(cell_wca.totnsegs)]
        [ax_m.plot(cell_wca.xmid[idx], cell_wca.zmid[idx], 'o',
                   c=idx_clr[idx], ms=13) for idx in plot_idxs]

        [ax_m.plot(cell_wca.xmid[idx], cell_wca.zmid[idx], 'rd')
         for idx in cell_wca.synidx]

        ax_top = 0.98
        ax_h = 0.25
        h_space = 0.1
        ax_w = 0.17
        ax_left = 0.4
        cell = cell_wca

        elec = LFPy.RecExtElectrode(cell, **electrode_parameters)
        elec.calc_lfp()

        ax_vm = fig.add_axes([ax_left, ax_top - ax_h, ax_w, ax_h],
                           ylim=[-80, 50], xlim=[0, 80],xlabel="Time (ms)")

        ax_eap = fig.add_axes([ax_left + 0.3, 0.1, ax_w, 0.8],
                           xlim=[0, 80], xlabel="Time (ms)")
        ax_cdm = fig.add_axes([ax_left, 0.2, ax_w, ax_h],
                              xlabel="Time (ms)", ylim=[-0.5, 1],
                              xlim=[0, 80],)

        ax_vm.set_ylabel("Membrane\npotential\n(mV)", labelpad=-3)
        ax_eap.set_ylabel("Extracellular potential ($\mu$V)", labelpad=-3)
        ax_cdm.set_ylabel("Curent dipole\nmoment\n($\mu$A$\cdot \mu$m)",
                          labelpad=-3)

        [ax_vm.plot(cell.tvec, cell.vmem[idx], c=idx_clr[idx])
         for idx in plot_idxs]

        elec.LFP -= elec.LFP[:, 0, None]
        cell.current_dipole_moment -= cell.current_dipole_moment[0, :]
        summed_LFP.append(elec.LFP)
        summed_cdm.append(cell.current_dipole_moment)

        normalize = np.max(np.abs(elec.LFP))
        for idx in range(len(elec_x)):
            ax_eap.plot(cell.tvec, elec.LFP[idx] / normalize * h + elec_z[idx],
                        c=elec_clr(idx))
            ax_m.plot(elec_x[idx], elec_z[idx], c=elec_clr(idx), marker='D')

        ax_cdm.plot(cell.tvec, 1e-3 * cell.current_dipole_moment[:, 2], c='k')

        mark_subplots([ax_m], xpos=0.1, ypos=0.95)
        simplify_axes(fig.axes)

        plt.savefig(join("figures", 'laminar_LFP_ca_spike_{}_{}.png'.format(
            cell_name, sim)))
        # plt.savefig(join("figures", 'hay_ca_spike.pdf'))
        plt.close("all")

    summed_LFP = np.sum(summed_LFP, axis=0)
    summed_cdm = np.sum(summed_cdm, axis=0)
    normalize = np.max(np.abs(summed_LFP))
    plt.subplot(121)
    for idx in range(len(elec_x)):
        plt.plot(cell.tvec, summed_LFP[idx] / normalize * h + elec_z[idx],
                 c=elec_clr(idx))

    plt.subplot(122)
    plt.plot(cell.tvec, summed_cdm[:, 2])

    plt.savefig(join("figures", 'summed_LFP_CDM_{}_num:{}.png'.format(
        cell_name, num_sims)))


def sum_jittered_cdm(cdm, dt, jitter_std, num_trials):

    cdm_len = len(cdm)

    tot_len = 10 * cdm_len
    summed_cdm = np.zeros(tot_len)

    for trial in range(num_trials):
        jitter_idxs = int(np.random.normal(0, jitter_std / dt))
        t0 = int(tot_len / 2 + jitter_idxs)
        t1 = int(t0 + cdm_len)
        summed_cdm[t0:t1] += (cdm - cdm[0])

    sum_tvec = np.arange(len(summed_cdm)) * dt
    sum_tvec -= sum_tvec[int(tot_len / 2)]
    summed_cdm -= summed_cdm[0]
    return sum_tvec, summed_cdm

if __name__ == '__main__':

    simulate_spike_current_dipole_moment()
    # animate_ca_spike()
    # simulate_laminar_LFP()

