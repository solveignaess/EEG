import os
from os.path import join
import numpy as np
from plotting_convention import mark_subplots, simplify_axes
import matplotlib.pyplot as plt
import neuron
import LFPy


def run_cell_simulation(make_ca_spike, dt):

    T = 100
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

    delay = 10
    synapse_s = LFPy.Synapse(cell, idx=cell.get_closest_idx(z=0),
                             syntype='Exp2Syn', weight=0.07, tau1=0.1, tau2=1.)
    synapse_s.set_spike_times(np.array([delay]))

    if make_ca_spike:
        synapse_a = LFPy.Synapse(cell, idx=cell.get_closest_idx(z=400),
                                 syntype='Exp2Syn', weight=0.05, tau1=0.1, tau2=2.)
        synapse_a.set_spike_times(np.array([delay]))
    cell.simulate(rec_imem=True, rec_vmem=True, rec_current_dipole_moment=True)

    return cell, idx_clr, plot_idxs

def simulate_hay_spike_current_dipole_moment():

    dt = 2**-5

    jitter_std = 10
    num_trials = 100

    elec_x = np.array([30,])
    elec_y = np.array([0, ])
    elec_z = np.array([0, ])

    electrode_parameters = {
        'sigma': 0.3,  # extracellular conductivity
        'x': elec_x,  # x,y,z-coordinates of contact points
        'y': elec_y,
        'z': elec_z
    }

    elec_clr = 'r'

    cell_woca, idx_clr, plot_idxs = run_cell_simulation(make_ca_spike=False, dt=dt)
    cell_wca, idx_clr, plot_idxs = run_cell_simulation(make_ca_spike=True, dt=dt)

    fig = plt.figure(figsize=[7, 7])
    fig.subplots_adjust(hspace=0.5, left=0.0, wspace=0.4, right=0.96,
                        top=0.97, bottom=0.1)

    ax_m = fig.add_axes([-0.01, 0.05, 0.3, 0.97], aspect=1, frameon=False,
                        xticks=[], yticks=[])
    [ax_m.plot([cell_wca.xstart[idx], cell_wca.xend[idx]],
              [cell_wca.zstart[idx], cell_wca.zend[idx]], c='k')
     for idx in range(cell_wca.totnsegs)]
    [ax_m.plot(cell_wca.xmid[idx], cell_wca.zmid[idx], 'o',
               c=idx_clr[idx], ms=13) for idx in plot_idxs]
    ax_m.plot(elec_x, elec_z, elec_clr, marker='D')

    ax_top = 0.98
    ax_h = 0.15
    h_space = 0.1
    ax_w = 0.2
    ax_left = 0.42

    for plot_row, cell in enumerate([cell_woca, cell_wca]):
        ax_left += plot_row * 0.3

        elec = LFPy.RecExtElectrode(cell, **electrode_parameters)
        elec.calc_lfp()

        sum_tvec, summed_cdm = sum_jittered_cdm(cell.current_dipole_moment[:, 2],
                                            dt, jitter_std, num_trials)

        ax_vm = fig.add_axes([ax_left, ax_top - ax_h, ax_w, ax_h],
                           ylim=[-80, 50], xlim=[0, 80],xlabel="Time (ms)")


        ax_eap = fig.add_axes([ax_left, ax_top - 2 * ax_h - h_space, ax_w, ax_h],
                           ylim=[-120, 40], xlim=[0, 80],xlabel="Time (ms)")
        ax_cdm = fig.add_axes([ax_left, ax_top - 3 * ax_h - 2*h_space, ax_w, ax_h],
                              xlabel="Time (ms)", ylim=[-0.5, 1],
                              xlim=[0, 80],)

        ax_cdm_sum = fig.add_axes([ax_left, ax_top - 4 * ax_h - 3*h_space, ax_w, ax_h],
                              ylim=[-20, 10],
                              xlabel="Time (ms)", xlim=[-20, 120])
        if plot_row == 0:
            ax_vm.set_ylabel("Membrane\npotential\n(mV)", labelpad=-3)
            ax_eap.set_ylabel("Extracellular\npotential\n($\mu$V)", labelpad=-3)
            ax_cdm.set_ylabel("Curent dipole\nmoment\n($\mu$A$\cdot \mu$m)", labelpad=-3)
            ax_cdm_sum.set_ylabel("Jittered sum\n($\mu$A$\cdot \mu$m)", labelpad=-3)
        mark_subplots(ax_vm, "BC"[plot_row], xpos=-0.3, ypos=0.93)
        [ax_vm.plot(cell.tvec, cell.vmem[idx], c=idx_clr[idx]) for idx in plot_idxs]
        # ax_cdm_sum = fig.add_subplot(524, ylim=[-1.1, 1.1], xlim=[0, 80],
        #                       ylabel="Membrane\ncurrents\n(normalized)")
        # [ax_cdm_sum.plot(cell.tvec, cell.imem[idx] / np.max(np.abs(cell.imem[idx])), c=idx_clr[idx])
        #  for idx in plot_idxs]


        [ax_eap.plot(cell.tvec, 1000 * elec.LFP[idx], c=elec_clr) for idx in range(len(elec_x))]

        ax_cdm.plot(cell.tvec, 1e-3 * cell.current_dipole_moment[:, 2], c='k')
        ax_cdm_sum.plot(sum_tvec, 1e-3 * summed_cdm, c='k')

    mark_subplots([ax_m], xpos=0.1, ypos=0.95)
    simplify_axes(fig.axes)

    plt.savefig(join("figures", 'hay_ca_spike.png'))
    plt.savefig(join("figures", 'hay_ca_spike.pdf'))


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
    simulate_hay_spike_current_dipole_moment()
