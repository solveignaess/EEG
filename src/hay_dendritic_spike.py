import os
from os.path import join
import numpy as np
from plotting_convention import mark_subplots, simplify_axes
import matplotlib.pyplot as plt
import neuron
import LFPy


def simulate_hay_spike_current_dipole_moment():
    T = 150
    dt = 2**-5
    model_folder = join('cell_models', 'hay', 'L5bPCmodelsEH')

    delay = 5
    make_ca_spike = False

    neuron.load_mechanisms(join(model_folder, "mod"))
    model_type = 'hay'

    ##define cell parameters used as input to cell-class
    cellParameters = {
        'morphology'    : join(model_folder, "morphologies", "cell1.asc"),
        'templatefile'  : [join(model_folder, "models", "L5PCbiophys3.hoc"),
                           join(model_folder, "models", "L5PCtemplate.hoc")],
        'templatename'  : 'L5PCtemplate',
        'templateargs'  : join(model_folder, "morphologies", "cell1.asc"),
        'passive' : False,
        'nsegs_method' : None,
        'dt' : dt,
        'tstart' : -159,
        'tstop' : T,
        'v_init' : -60,
        'celsius': 34,
        'pt3d' : True,
    }


    cell = LFPy.TemplateCell(**cellParameters)

    cell.set_rotation(x=np.pi/2, y=0.1)

    plot_idxs = [cell.somaidx[0],
                 #cell.get_closest_idx(z=200),
                 cell.get_closest_idx(z=500)]
    idx_clr = {idx: plt.cm.viridis(num / len(plot_idxs)) for num, idx in enumerate(plot_idxs)}


    synapse_s = LFPy.Synapse(cell, idx=cell.get_closest_idx(z=0),
                             syntype='Exp2Syn', weight=0.07, tau1=0.1, tau2=1.)
    synapse_s.set_spike_times(np.array([delay]))

    if make_ca_spike:
        synapse_a = LFPy.Synapse(cell, idx=cell.get_closest_idx(z=400),
                                 syntype='Exp2Syn', weight=0.05, tau1=0.1, tau2=2.)
        synapse_a.set_spike_times(np.array([delay]))

    cell.simulate(rec_imem=True, rec_vmem=True, rec_current_dipole_moment=True)


    np.save('hay_cdm_with{}_ca_spike.npy'.format("" if make_ca_spike else "out"),
            cell.current_dipole_moment[:, 2])

    elec_x = np.array([30,])
    elec_y = np.array([0, ])
    elec_z = np.array([0, ])

    electrode_parameters = {
        'sigma': 0.3,  # extracellular conductivity
        'x': elec_x,  # x,y,z-coordinates of contact points
        'y': elec_y,
        'z': elec_z
    }
    elec = LFPy.RecExtElectrode(cell, **electrode_parameters)
    elec.calc_lfp()
    elec_clr = 'r'


    fig = plt.figure(figsize=[5, 7])
    fig.subplots_adjust(hspace=0.5, left=-0.0, wspace=0.4, right=0.98,
                        top=0.97, bottom=0.1)

    ax1 = fig.add_subplot(422, ylim=[-80, 50], xlim=[0, 80],
                          ylabel="Membrane\npotential\n(mv)")
    [ax1.plot(cell.tvec, cell.vmem[idx], c=idx_clr[idx]) for idx in plot_idxs]

    ax2 = fig.add_subplot(424, ylim=[-1.1, 1.1], xlim=[0, 80],
                          ylabel="Membrane\ncurrents\n(normalized)")
    [ax2.plot(cell.tvec, cell.imem[idx] / np.max(np.abs(cell.imem[idx])), c=idx_clr[idx])
     for idx in plot_idxs]

    ax3 = fig.add_subplot(426, ylim=[-120, 40], xlim=[0, 80],
                          ylabel="Extracellular\npotential\n($\mu$V)",)
    [ax3.plot(cell.tvec, 1000 * elec.LFP[idx], c=elec_clr) for idx in range(len(elec_x))]

    ax4 = fig.add_subplot(428, ylabel="Curent dipole\nmoment\n($\mu$A$\cdot \mu$m)",
                          xlabel="Time (ms)", ylim=[-0.5, 1], xlim=[0, 80],)
    ax4.plot(cell.tvec, 1e-3 * cell.current_dipole_moment[:, 2], c='k')

    ax_m = fig.add_axes([0.01, 0.05, 0.3, 0.97], aspect=1, frameon=False,
                        xticks=[], yticks=[])
    [ax_m.plot([cell.xstart[idx], cell.xend[idx]],
              [cell.zstart[idx], cell.zend[idx]], c='k') for idx in range(cell.totnsegs)]
    [ax_m.plot(cell.xmid[idx], cell.zmid[idx], 'o', c=idx_clr[idx], ms=13) for idx in plot_idxs]
    ax_m.plot(elec_x, elec_z, elec_clr, marker='D')

    simplify_axes(fig.axes)

    plt.savefig('hay_EAP_test_with{}_ca_spike.png'.format("" if make_ca_spike else "out"))

def sum_current_dipole_moments():

    dt = 2**-5
    cdm_with_ca = np.load('hay_cdm_with_ca_spike.npy')
    cdm_without_ca = np.load('hay_cdm_without_ca_spike.npy')

    cdm_with_ca -= cdm_with_ca[0]
    cdm_without_ca -= cdm_without_ca[0]

    cdm_len = len(cdm_with_ca)
    tvec = np.arange(cdm_len) * dt
    jitter_std = 10
    num_trials = 100

    tot_len = 10 * cdm_len
    summed_cdm_with_ca = np.zeros(tot_len)
    summed_cdm_without_ca = np.zeros(tot_len)
    summed_tvec = np.arange(tot_len) * dt
    summed_tvec -= summed_tvec[int(tot_len/2)]

    plt.close("all")
    fig = plt.figure(figsize=[6, 7])
    fig.subplots_adjust(left=0.22, right=0.98, top=0.94)

    ax1 = fig.add_subplot(221, xlim=[-50, 170], ylim=[-0.3, 1.0],
                          title="without Ca spike",
                          ylabel="Single cell P$_z$\n($\mu$A$\cdot \mu$m)")

    ax2 = fig.add_subplot(223, xlim=[-50, 170], ylim=[-20, 10],
                      ylabel="{} summed intances\nwith {} ms jitter\n($\mu$A$\cdot \mu$m)".format(num_trials, jitter_std),
                      xlabel="Time (ms)", )

    ax3 = fig.add_subplot(222, xlim=[-50, 170], ylim=[-0.3, 1.0],
                          title="with Ca spike")

    ax4 = fig.add_subplot(224, xlim=[-50, 170], ylim=[-20, 10],
                      #title="{} summed intances\nwith {} ms jitter".format(num_trials, jitter_std),
                      xlabel="Time (ms)")

    ax1.plot(tvec, cdm_without_ca / 1000)
    ax1.axhline(0, ls='--', c='gray')

    ax3.plot(tvec, cdm_with_ca / 1000)
    ax3.axhline(0, ls='--', c='gray')

    for trial in range(num_trials):
        jitter_idxs = int(np.random.normal(0, jitter_std / dt))
        # print(jitter_idxs, jitter_idxs * dt)
        t0 = int(tot_len / 2 + jitter_idxs)
        t1 = int(t0 + cdm_len)
        tmp_sig_with_ca = np.zeros(tot_len)
        tmp_sig_without_ca = np.zeros(tot_len)

        tmp_sig_with_ca[t0:t1] = cdm_with_ca
        tmp_sig_without_ca[t0:t1] = cdm_without_ca
        ax2.plot(summed_tvec, tmp_sig_without_ca / 1000, 'gray')
        ax4.plot(summed_tvec, tmp_sig_with_ca / 1000, 'gray')
        summed_cdm_without_ca[t0:t1] += cdm_without_ca
        summed_cdm_with_ca[t0:t1] += cdm_with_ca

    ax2.plot(summed_tvec, summed_cdm_without_ca / 1000, c='k')
    ax4.plot(summed_tvec, summed_cdm_with_ca / 1000, c='k')
    simplify_axes(fig.axes)
    plt.savefig("jittered_dendritic_spike_cdm.png")

if __name__ == '__main__':
    # simulate_hay_spike_current_dipole_moment()
    sum_current_dipole_moments()