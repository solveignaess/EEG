import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
if sys.version < '3':
    from urllib2 import urlopen
else:
    from urllib.request import urlopen
import ssl


class NYHeadModel:
    """
    Main class for New York head model
    Huang Y, Parra LC, Haufe S (2016) Neuroimage 140:150–162.
    Assumes units of nA * um for current dipole moment, and pV for EEG
    NOTE: The original unit of the New York model current dipole moment
    is (probably?) mA * m, and the EEG output is V
    LFPy's current dipole moments have units nA*um, giving EEGs in pV.
    """

    def __init__(self, num_tsteps=None, dt=1):

        self.root_folder = "."
        self.illustrations_folder = join(self.root_folder, "head_model_illustrations")
        self.head_file = join(self.root_folder, "sa_nyhead.mat")
        if not os.path.isfile(self.head_file):
            print("New York head model not found in: %s" % self.head_file)
            yn = input("Should it be downloaded (710 MB)? [y/n]")
            if yn == 'y':
                print("Now downloading. This might take a while ...")
                nyhead_url = 'https://www.parralab.org/nyhead/sa_nyhead.mat'
                u = urlopen(nyhead_url, context=ssl._create_unverified_context())
                localFile = open(self.head_file, 'wb')
                localFile.write(u.read())
                localFile.close()
                print("Download done!")
            else:
                print("Exiting program ...")
                sys.exit()

        self.head_data = h5py.File(self.head_file, 'r')["sa"]

        self.x_lim = [-100, 100]
        self.y_lim = [-130, 100]
        self.z_lim = [-160, 120]

        # self.cortex = np.array(self.head_data["cortex75K"]["vc_smooth"])
        self.cortex = np.array(self.head_data["cortex75K"]["vc"])
        self.lead_field = np.array(self.head_data["cortex75K"]["V_fem"])
        self.lead_field_normal = np.array(self.head_data["cortex75K"]["V_fem_normal"])
        # self.sulicmap = np.array(f["sa"]["cortex75K"]["sulcimap"])[0,:]
        self.cortex_normals = np.array(self.head_data["cortex75K"]["normals"])
        self.elecs = np.array(self.head_data["locs_3D"])
        self.head = np.array(self.head_data["head"]["vc"])
        self.head_tri = np.array(self.head_data["head"]["tri"], dtype=int) - 1
        self.cortex_tri = np.array(self.head_data["cortex75K"]["tri"], dtype=int)[:, :] - 1
        # elecs2D = np.array(f["sa"]["locs_2D"])

        self.num_elecs = self.elecs.shape[1]
        print(self.num_elecs)
        self.num_tsteps = num_tsteps
        self.dt = dt
        if not None in [dt, num_tsteps]:
            self.t = np.arange(self.num_tsteps) * dt

        self.dipole_pos_dict = {
            'calcarine_sulcus': np.array([5, -85, 0]),
            'motorsensory_cortex': np.array([17, 10, 79.4]),
            'back_of_head': np.array([-14.3, -99.6, 22.0]),
            'parietal_lobe': np.array([55, -49, 57]),
            'occipital_lobe': np.array([-24.3, -105.4, -1.2])
        }

    def make_dipole_timecourse(self, t0=0):
        t0_idx = np.argmin(np.abs(self.t - t0))
        # np.random.seed(1234)
        # dipole_moment = np.random.normal(0, 100, size=(3, self.num_tsteps))
        dipole_moment = np.zeros((3, self.num_tsteps))

        dipole_moment[2, t0_idx:] += 1000 * np.exp(-self.t[:self.num_tsteps - t0_idx])  # Units nA um
        self.set_dipole_moment(dipole_moment)

    def load_hybrid_current_dipole(self):
        cdm_file = join('..', "hybrid_EEG_evoked", "evoked_cdm", "summed_cdm.npy")
        cdm = np.load(cdm_file).T
        self.set_dipole_moment(cdm)

    def plot_dipole_timecorse(self, figname="cdm.png"):
        plt.close("all")
        plt.plot(self.t, self.dipole_moment[0, :])
        plt.plot(self.t, self.dipole_moment[1, :])
        plt.plot(self.t, self.dipole_moment[2, :])
        plt.savefig(join(figname))

    def set_dipole_pos(self, dipole_pos_0=None):

        if dipole_pos_0 is None:
            dipole_pos_0 = self.dipole_pos_dict['motorsensory_cortex']
        if type(dipole_pos_0) is str:
            self.closest_vertex_idx = self.return_closest_idx(
                self.dipole_pos_dict[dipole_pos_0])
        else:
            self.closest_vertex_idx = self.return_closest_idx(dipole_pos_0)
        self.dipole_pos = self.cortex[:, self.closest_vertex_idx]
        print("Dipole pos: ", self.dipole_pos)
        print("Normal vector: ", self.cortex_normals[:, self.closest_vertex_idx])

    def set_dipole_moment(self, dipole_moment=np.array([[0], [0], [500]])):
        """
        dipole_moment: (3 * num_timesteps) numpy array
        Units: nA * um
        """
        if dipole_moment.shape == (3,):
            self.dipole_moment = np.array([dipole_moment]).T
            self.num_tsteps = self.dipole_moment.shape[1]
        elif dipole_moment.shape[0] == 3:
            self.dipole_moment = dipole_moment
            self.num_tsteps = dipole_moment.shape[1]
            self.t = np.arange(self.num_tsteps) * self.dt
        else:
            raise NotImplementedError()
        # print("P:", dipole_moment)

    def rotate_dipole_moment(self):
        """
        rotate dipole moment vector such that it points in the direction of
        the brain surface normal.
        We assume that the pyramidal cells generating the dipole moments is
        aligned with the z-axis.

        For use when applying lead_field.
        """
        p_length = self.dipole_moment[2]
        p_idx = self.return_closest_idx(self.dipole_pos)
        print('p_idx:', p_idx)
        n = self.cortex_normals[:, p_idx]
        print('n:', n)
        dipole_moment_rot = np.outer(n, p_length)
        print('dipole_moment_rot:', dipole_moment_rot)
        return dipole_moment_rot

    def find_closest_electrode(self):
        dists = (np.sqrt(np.sum((np.array(self.dipole_pos)[:, None] -
                                 np.array(self.elecs[:3, :]))**2, axis=0)))
        closest_electrode = np.argmin(dists)
        min_dist = np.min(dists)
        return min_dist, closest_electrode

    def calculate_eeg_signal(self, normal=True):
        self.eeg = np.zeros((self.num_elecs, self.num_tsteps))

        if normal:
            self.eeg[:, :] = self.lead_field_normal[None, self.closest_vertex_idx, :].T @  self.dipole_moment[None, 2, :]
        else:
            dipole_moment_rot = self.rotate_dipole_moment()

            # for tstep in range(self.num_tsteps):
                # for elec in range(self.num_elecs):
                #     eeg_ = np.dot(self.lead_field[:, self.closest_vertex_idx, elec].T,
                #                   dipole_moment_rot[:, tstep])
            self.eeg[:, :] = self.lead_field[:, self.closest_vertex_idx, :].T @ dipole_moment_rot[:, :]
            print("no lead_field_normal")
        print('no conversion of eeg')
        print("Max 4o EEG amp: {:2.02f}".format(np.max(np.abs(self.eeg[:, 0:]))))

    def return_closest_idx(self, pos):
        return np.argmin((self.cortex[0, :] - pos[0])**2 +
                         (self.cortex[1, :] - pos[1])**2 +
                         (self.cortex[2, :] - pos[2])**2)

    def return_each_electrode_max_amplitude(self):
        high_amp_idxs = np.argmax(np.abs(self.lead_field_normal), axis=0)
        return high_amp_idxs

    def plot_field_and_crossection(self, fig_name):
        plt.close("all")
        fig = plt.figure(figsize=[15, 9.5])
        fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.17,
                            wspace=0.05, left=0.1, right=0.99)
        ax1 = fig.add_subplot(234, aspect=1, xlabel="x (mm)", ylabel='y (mm)',
                              xlim=self.x_lim, ylim=self.y_lim)
        ax2 = fig.add_subplot(235, aspect=1, xlabel="x (mm)", ylabel='z (mm)',
                              xlim=self.x_lim, ylim=self.z_lim)
        ax3 = fig.add_subplot(236, aspect=1, xlabel="y (mm)", ylabel='z (mm)',
                              xlim=self.y_lim, ylim=self.z_lim)
        max_elec_idx = np.argmax(np.std(self.eeg, axis=1))
        time_idx = np.argmax(np.abs(self.eeg[max_elec_idx]))
        max_eeg = np.max(np.abs(self.eeg[:, time_idx]))
        max_eeg_idx = np.argmax(np.abs(self.eeg[:, time_idx]))
        max_eeg_pos = self.elecs[:3, max_eeg_idx]
        fig.text(0.01, 0.25, "Cortex", va='center', rotation=90, fontsize=22)
        fig.text(0.03, 0.25, "Dipole pos: {:1.1f}, {:1.1f}, {:1.1f}\nDipole moment: {:1.2f} {:1.2f} {:1.2f}".format(
            self.dipole_pos[0], self.dipole_pos[1], self.dipole_pos[2],
            self.dipole_moment[0, time_idx], self.dipole_moment[1, time_idx], self.dipole_moment[2, time_idx]
        ), va='center', rotation=90, fontsize=14)

        fig.text(0.01, 0.75,
             "EEG", va='center', rotation=90, fontsize=22)
        fig.text(0.03, 0.75,
             "Max: {:1.2f} $\mu$V at idx {}\n({:1.1f}, {:1.1f} {:1.1f})".format(
                 max_eeg, max_eeg_idx, max_eeg_pos[0], max_eeg_pos[1], max_eeg_pos[2]),
                 va='center', rotation=90, fontsize=14)

        ax7 = fig.add_subplot(231, aspect=1, xlabel="x (mm)", ylabel='y (mm)',
                              xlim=self.x_lim, ylim=self.y_lim)
        ax8 = fig.add_subplot(232, aspect=1, xlabel="x (mm)", ylabel='z (mm)',
                              xlim=self.x_lim, ylim=self.z_lim)
        ax9 = fig.add_subplot(233, aspect=1, xlabel="y (mm)", ylabel='z (mm)',
                              xlim=self.y_lim, ylim=self.z_lim)

        # [ax.grid(True) for ax in [ax1, ax2, ax3, ax7, ax8, ax9]]

        vmax = np.max(np.abs(self.eeg[:, time_idx]))
        v_range = vmax
        cmap = lambda v: plt.cm.bwr((v + vmax) / (2*vmax))
        # cmap = np.vectorize(plt.cm.viridis())

        threshold = 2

        xz_plane_idxs = np.where(np.abs(self.cortex[1, :] - self.dipole_pos[1]) < threshold)[0]
        xy_plane_idxs = np.where(np.abs(self.cortex[2, :] - self.dipole_pos[2]) < threshold)[0]
        yz_plane_idxs = np.where(np.abs(self.cortex[0, :] - self.dipole_pos[0]) < threshold)[0]

        # ax1.scatter(cortex[0, xz_plane_idxs], cortex[2, xz_plane_idxs])

        ax1.scatter(self.cortex[0, xy_plane_idxs], self.cortex[1, xy_plane_idxs], s=5)
        ax2.scatter(self.cortex[0, xz_plane_idxs], self.cortex[2, xz_plane_idxs], s=5)
        ax3.scatter(self.cortex[1, yz_plane_idxs], self.cortex[2, yz_plane_idxs], s=5)

        ax1.scatter(self.dipole_pos[0], self.dipole_pos[1], s=15, color='orange')
        ax2.scatter(self.dipole_pos[0], self.dipole_pos[2], s=15, color='orange')
        ax3.scatter(self.dipole_pos[1], self.dipole_pos[2], s=15, color='orange')
        img = plt.imshow([[], []], origin="lower", vmin=-vmax,
                         vmax=vmax, cmap=plt.cm.bwr)
        plt.colorbar(img)

        ax7.scatter(self.elecs[0,:], self.elecs[1,:], s=50,
                    c=cmap(self.eeg[:, time_idx]))
        ax8.scatter(self.elecs[0,:], self.elecs[2,:], s=50, c=cmap(self.eeg[:, time_idx]))
        ax9.scatter(self.elecs[1,:], self.elecs[2,:], s=50, c=cmap(self.eeg[:, time_idx]))

        ax7.scatter(self.dipole_pos[0], self.dipole_pos[1], s=15, color='orange')
        ax8.scatter(self.dipole_pos[0], self.dipole_pos[2], s=15, color='orange')
        ax9.scatter(self.dipole_pos[1], self.dipole_pos[2], s=15, color='orange')

        plt.savefig(join(self.root_folder, fig_name))

    def plot_EEG_results(self, fig_name):
        plt.close("all")
        fig = plt.figure(figsize=[19, 10])
        fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.17, wspace=0.3, left=0.1, right=0.99)
        ax1 = fig.add_subplot(245, aspect=1, xlabel="x (mm)", ylabel='y (mm)', xlim=self.x_lim, ylim=self.y_lim)
        ax2 = fig.add_subplot(246, aspect=1, xlabel="x (mm)", ylabel='z (mm)', xlim=self.x_lim, ylim=self.z_lim)
        ax3 = fig.add_subplot(247, aspect=1, xlabel="y (mm)", ylabel='z (mm)', xlim=self.y_lim, ylim=self.z_lim)
        ax_eeg = fig.add_subplot(244, xlabel="Time (ms)", ylabel='$p$V', title='EEG')

        ax_cdm = fig.add_subplot(248, xlabel="Time (ms)", ylabel='nA$\cdot \mu$m', title='Current dipole moment')
        dist, closest_elec_idx = head.find_closest_electrode()
        print("Closest electrode to dipole: {:1.2f} mm".format(dist))
        max_elec_idx = np.argmax(np.std(self.eeg, axis=1))
        time_idx = np.argmax(np.abs(self.eeg[max_elec_idx]))
        max_eeg = np.max(np.abs(self.eeg[:, time_idx]))
        max_eeg_idx = np.argmax(np.abs(self.eeg[:, time_idx]))
        max_eeg_pos = self.elecs[:3, max_eeg_idx]
        fig.text(0.01, 0.25, "Cortex", va='center', rotation=90, fontsize=22)
        fig.text(0.03, 0.25, "Dipole pos: {:1.1f}, {:1.1f}, {:1.1f}\nDipole moment: {:1.2f} {:1.2f} {:1.2f}".format(
            self.dipole_pos[0], self.dipole_pos[1], self.dipole_pos[2],
            self.dipole_moment[0, time_idx], self.dipole_moment[1, time_idx], self.dipole_moment[2, time_idx]
        ), va='center', rotation=90, fontsize=14)

        fig.text(0.01, 0.75, "EEG", va='center', rotation=90, fontsize=22)
        fig.text(0.03, 0.75, "Max: {:1.2f} pV at idx {}\n({:1.1f}, {:1.1f} {:1.1f})".format(
                 max_eeg, max_eeg_idx, max_eeg_pos[0], max_eeg_pos[1], max_eeg_pos[2]), va='center',
                 rotation=90, fontsize=14)

        ax7 = fig.add_subplot(241, aspect=1, xlabel="x (mm)", ylabel='y (mm)',
                              xlim=self.x_lim, ylim=self.y_lim)
        ax8 = fig.add_subplot(242, aspect=1, xlabel="x (mm)", ylabel='z (mm)',
                              xlim=self.x_lim, ylim=self.z_lim)
        ax9 = fig.add_subplot(243, aspect=1, xlabel="y (mm)", ylabel='z (mm)',
                              xlim=self.y_lim, ylim=self.z_lim)

        ax_cdm.plot(self.t, self.dipole_moment[2, :], 'k')
        [ax_eeg.plot(self.t, self.eeg[idx, :], c='gray') for idx in range(self.num_elecs)]
        ax_eeg.plot(self.t, self.eeg[closest_elec_idx, :], c='green', lw=2)
        # print(self.eeg[50, :])

        vmax = np.max(np.abs(self.eeg[:, time_idx]))
        v_range = vmax
        cmap = lambda v: plt.cm.bwr((v + vmax) / (2*vmax))
        # cmap = np.vectorize(plt.cm.viridis())

        threshold = 2

        xz_plane_idxs = np.where(np.abs(self.cortex[1, :] - self.dipole_pos[1]) < threshold)[0]
        xy_plane_idxs = np.where(np.abs(self.cortex[2, :] - self.dipole_pos[2]) < threshold)[0]
        yz_plane_idxs = np.where(np.abs(self.cortex[0, :] - self.dipole_pos[0]) < threshold)[0]

        ax1.scatter(self.cortex[0, xy_plane_idxs], self.cortex[1, xy_plane_idxs], s=5)
        ax2.scatter(self.cortex[0, xz_plane_idxs], self.cortex[2, xz_plane_idxs], s=5)
        ax3.scatter(self.cortex[1, yz_plane_idxs], self.cortex[2, yz_plane_idxs], s=5)

        ax1.scatter(self.dipole_pos[0], self.dipole_pos[1], s=15, color='orange')
        ax2.scatter(self.dipole_pos[0], self.dipole_pos[2], s=15, color='orange')
        ax3.scatter(self.dipole_pos[1], self.dipole_pos[2], s=15, color='orange')

        ax7.scatter(self.elecs[0,:], self.elecs[1,:], s=50,
                    c=cmap(self.eeg[:, time_idx]))
        ax8.scatter(self.elecs[0,:], self.elecs[2,:], s=50, c=cmap(self.eeg[:, time_idx]))
        ax9.scatter(self.elecs[1,:], self.elecs[2,:], s=50, c=cmap(self.eeg[:, time_idx]))

        img = ax3.imshow([[], []], origin="lower", vmin=-vmax,
                         vmax=vmax, cmap=plt.cm.bwr)
        plt.colorbar(img, ax=ax9, shrink=0.5)

        ax7.scatter(self.dipole_pos[0], self.dipole_pos[1], s=15, color='orange')
        ax8.scatter(self.dipole_pos[0], self.dipole_pos[2], s=15, color='orange')
        ax9.scatter(self.dipole_pos[1], self.dipole_pos[2], s=15, color='orange')

        plt.savefig(join(self.root_folder, fig_name))

    def plot_hybrid_current_dipole(self):
        self.load_hybrid_current_dipole()
        self.set_dipole_pos('back_of_head')
        self.calculate_eeg_signal()
        eeg_elec = np.argmax(np.max(np.abs(self.eeg[:, 200:]), axis=1))
        plt.plot(self.t, self.eeg[eeg_elec, :])
        plt.show()

    def plot_head_model(self):

        fig = plt.figure(figsize=[9.4, 9.5])
        fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.4,
                            wspace=0.4, left=0.055, right=0.99)
        ax1 = fig.add_subplot(331, aspect=1, xlabel="x (mm)", ylabel='y (mm)',
                              title='Cortex')
        ax2 = fig.add_subplot(332, aspect=1, xlabel="x (mm)", ylabel='z (mm)',
                              title='Cortex')
        ax3 = fig.add_subplot(333, aspect=1, xlabel="y (mm)", ylabel='z (mm)',
                              title='Cortex')
        ax4 = fig.add_subplot(334, aspect=1, xlabel="x (mm)", ylabel='y (mm)',
                              title='Head')
        ax5 = fig.add_subplot(335, aspect=1, xlabel="x (mm)", ylabel='z (mm)',
                              title='Head')
        ax6 = fig.add_subplot(336, aspect=1, xlabel="y (mm)", ylabel='z (mm)',
                              title='Head')
        ax7 = fig.add_subplot(337, aspect=1, xlabel="x (mm)", ylabel='y (mm)',
                              title='EEG')
        ax8 = fig.add_subplot(338, aspect=1, xlabel="x (mm)", ylabel='z (mm)',
                              title='EEG')
        ax9 = fig.add_subplot(339, aspect=1, xlabel="y (mm)", ylabel='z (mm)',
                              title='EEG')

        ax1.scatter(self.cortex[0, :], self.cortex[1, :], s=2)
        ax2.scatter(self.cortex[0, :], self.cortex[2, :], s=2)
        ax3.scatter(self.cortex[1, :], self.cortex[2, :], s=2)
        ax4.scatter(self.head[0, :], self.head[1, :], s=2)
        ax5.scatter(self.head[0, :], self.head[2, :], s=2)
        ax6.scatter(self.head[1, :], self.head[2, :], s=2)
        ax7.scatter(self.elecs[0, :], self.elecs[1, :])
        ax8.scatter(self.elecs[0, :], self.elecs[2, :])
        ax9.scatter(self.elecs[1, :], self.elecs[2, :])
        plt.savefig("NY_head_model.png")

    def plot_lead_fields(self):
        x_lim = [-100, 100]
        y_lim = [-130, 100]
        z_lim = [-160, 120]

        fig_folder = join(self.root_folder, 'lead_fields')
        os.makedirs(fig_folder, exist_ok=True)

        for elec in range(self.lead_field_normal.shape[1]):
            print("Plotting lead field of elec number %d / %d" % (elec, self.num_elecs))
            plt.close("all")
            fig = plt.figure(figsize=[18, 9.5])
            fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.4,
                                wspace=0.4, left=0.055, right=0.99)

            ax7 = fig.add_subplot(231, aspect=1, xlabel="x (mm)", ylabel='y (mm)',
                                  title='EEG', xlim=x_lim, ylim=y_lim)
            ax8 = fig.add_subplot(232, aspect=1, xlabel="x (mm)", ylabel='z (mm)',
                                  title='EEG', xlim=x_lim, ylim=z_lim)
            ax9 = fig.add_subplot(233, aspect=1, xlabel="y (mm)", ylabel='z (mm)',
                                  title='EEG', xlim=y_lim, ylim=z_lim)

            ax1 = fig.add_subplot(234, aspect=1, xlabel="x (mm)", ylabel='y (mm)',
                                  title='Cortex', xlim=x_lim, ylim=y_lim)
            ax2 = fig.add_subplot(235, aspect=1, xlabel="x (mm)", ylabel='z (mm)',
                                  title='Cortex', xlim=x_lim, ylim=z_lim)
            ax3 = fig.add_subplot(236, aspect=1, xlabel="y (mm)", ylabel='z (mm)',
                                  title='Cortex', xlim=y_lim, ylim=z_lim)

            vmax = np.max(np.abs(self.lead_field[:, elec]))

            cmap = lambda v: plt.cm.bwr((v + vmax) / (2 * vmax))

            ax1.scatter(self.cortex[0, :], self.cortex[1, :],
                        c=cmap(self.lead_field_normal[:, elec]), s=1)
            ax2.scatter(self.cortex[0, :], self.cortex[2, :],
                        c=cmap(self.lead_field_normal[:, elec]), s=1)
            ax3.scatter(self.cortex[1, :], self.cortex[2, :],
                        c=cmap(self.lead_field_normal[:, elec]), s=1)
            img = plt.imshow([[], []], origin="lower", vmin=-vmax,
                             vmax=vmax,cmap=plt.cm.bwr)
            plt.colorbar(img)
            ax7.scatter(self.elecs[0,:], self.elecs[1,:], s=150)
            ax8.scatter(self.elecs[0,:], self.elecs[2,:], s=150)
            ax9.scatter(self.elecs[1,:], self.elecs[2,:], s=150)

            ax7.scatter(self.elecs[0,elec], self.elecs[1,elec], s=150)
            ax8.scatter(self.elecs[0,elec], self.elecs[2,elec], s=150)
            ax9.scatter(self.elecs[1,elec], self.elecs[2,elec], s=150)

            plt.savefig(join(fig_folder, "NY_lead_field_elec_{}.png".format(elec)))

    def plot_brain_crossections(self):
        thresold = 2
        fig_folder = join(self.root_folder, "crossections")
        os.makedirs(fig_folder, exist_ok=True)

        plot_xy_plane = False
        plot_zy_plane = False
        plot_xz_plane = True

        num_plot_slices = 100

        if plot_xy_plane:
            print("Plotting XY-plane.")
            for num, z_pos in enumerate(np.linspace(-50, 80, num_plot_slices)):
                print("%d / %d" % (num, num_plot_slices))
                xy_plane_idxs = np.where(np.abs(self.cortex[2, :] - z_pos) < thresold)[0]
                plt.close("all")
                fig = plt.figure(figsize=[18.4, 9.5])
                fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.4,
                                    wspace=0.4, left=0.055, right=0.99)
                ax1 = fig.add_subplot(111, aspect=1, xlabel="x (mm)", ylabel='y (mm)',
                                      xlim=[-80, 80], ylim=[-110, 80])
                ax1.grid(True)
                ax1.scatter(self.cortex[0, xy_plane_idxs], self.cortex[1, xy_plane_idxs])

                plt.savefig(join(fig_folder, "NY_crossection_xy_{:04d}_z{:0.2f}.png".format(num, z_pos)))

        if plot_zy_plane:
            print("Plotting ZY-plane.")
            for num, x_pos in enumerate(np.linspace(-80, 80, num_plot_slices)):
                print("%d / %d" % (num, num_plot_slices))
                zy_plane_idxs = np.where(np.abs(self.cortex[0, :] - x_pos) < thresold)[0]
                plt.close("all")
                fig = plt.figure(figsize=[18.4, 9.5])
                fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.4,
                                    wspace=0.4, left=0.055, right=0.99)
                ax1 = fig.add_subplot(111, aspect=1, xlabel="y (mm)", ylabel='z (mm)',
                                      ylim=[-50, 80], xlim=[-110, 80])
                ax1.grid(True)
                ax1.scatter(self.cortex[1, zy_plane_idxs], self.cortex[2, zy_plane_idxs])

                plt.savefig(join(fig_folder, "NY_crossection_yz_{:04d}_x{:0.2f}.png".format(num, x_pos)))

        if plot_xz_plane:
            print("Plotting XZ-plane.")
            for num, y_pos in enumerate(np.linspace(-110, 80, num_plot_slices)):
                print("%d / %d" % (num, num_plot_slices))
                xz_plane_idxs = np.where(np.abs(self.cortex[1, :] - y_pos) < thresold)[0]
                plt.close("all")
                fig = plt.figure(figsize=[18.4, 9.5])
                fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.4,
                                    wspace=0.4, left=0.055, right=0.99)
                ax1 = fig.add_subplot(111, aspect=1, xlabel="x (mm)", ylabel='z (mm)',
                                      ylim=[-70, 80], xlim=[-80, 80])
                ax1.grid(True)
                ax1.scatter(self.cortex[0, xz_plane_idxs], self.cortex[2, xz_plane_idxs])

                plt.savefig(join(fig_folder, "NY_crossection_xz_{:04d}_y{:0.2f}.png".format(num, y_pos)))

    def plot_all_elec_max_amps(self):

        fig_folder = "max_elec_amp"
        os.makedirs(join(head.root_folder, fig_folder), exist_ok=True)

        elecs_max_vertex_idxs = self.return_each_electrode_max_amplitude()

        for elec_idx in range(self.num_elecs):
            vertex_idx = elecs_max_vertex_idxs[elec_idx]
            head.set_dipole_pos(head.cortex[:, vertex_idx])
            head.set_dipole_moment(head.cortex_normals[:, vertex_idx])
            head.calculate_eeg_signal()
            fig_name = join(fig_folder, "NY_max_amp_elec:{}.png".format(elec_idx))
            head.plot_field_and_crossection(fig_name)

if __name__ == '__main__':
    head = NYHeadModel(num_tsteps=100, dt=0.1)
    # head.plot_hybrid_current_dipole()
    # head.plot_lead_fields()
    # head.plot_brain_crossections()
    # head.plot_head_model()
    # head.plot_all_elec_max_amps()
    head.set_dipole_pos()
    head.make_dipole_timecourse(t0=2)
    # head.set_dipole_moment()
    head.plot_dipole_timecorse()
    normal = False
    import time
    t0 = time.time()
    head.calculate_eeg_signal(normal=normal)
    t3 = time.time()
    eeg2 = head.eeg.copy()

    head.plot_EEG_results("test_EEG_results_normal:{}.png".format(normal))
    # head.plot_field_and_crossection("test_cross_section.png")
