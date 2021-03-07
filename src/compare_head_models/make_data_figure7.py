import numpy as np
import LFPy
import time
from main import NYHeadModel
from equidistal_surface_points import return_equidistal_xyz


def generate_eegs_nyh_n_4s(dip_loc):
    # set 4S properties
    # From Huang et al. (2013): 10.1088/1741-2560/10/6/066004
    sigmas = [0.276, 1.65, 0.01, 0.465]
    radii = [89000., 90000., 95000., 100000.]

    print('sigmas:', sigmas)
    rad_tol = 1e-2

    P_loc_4s_length = radii[0] - 1000
    x_eeg, y_eeg, z_eeg = return_equidistal_xyz(224, r=radii[-1] - rad_tol)
    eeg_coords_4s = np.array([x_eeg, y_eeg, z_eeg]).T

    # get population dipole
    nyh = NYHeadModel()
    # hybrid current dipole moment
    nyh.load_hybrid_current_dipole()
    P = nyh.dipole_moment[:, 875:951]
    # place dipole and find normal vector
    P_loc_nyh = nyh.dipole_pos_dict[dip_loc]

    nyh.set_dipole_pos(dipole_pos_0=P_loc_nyh)
    P_loc_idx = nyh.return_closest_idx(P_loc_nyh)
    norm_vec = nyh.cortex_normals[:, P_loc_idx]
    # find location on 4S
    # want to find position on brain, such that the location vector is
    # parallel to norm_vec from nyh.
    # length of position vector
    P_loc_4s = norm_vec*P_loc_4s_length
    P_rot = nyh.rotate_dipole_moment()[:, 875:951]
    # use this when comparing execution times

    elec_dists_4s = np.sqrt(np.sum((eeg_coords_4s - P_loc_4s)**2, axis=1))
    elec_dists_4s *= 1e-3 # convert from um to mm
    min_dist_idx = np.argmin(elec_dists_4s)
    print("Minimum 4o distance {:2.02f} mm".format(elec_dists_4s[min_dist_idx]))

    time_idx = np.argmax(np.linalg.norm(P_rot, axis=0))
    # if execution times:
    # time_idx = np.argmax(np.linalg.norm(P_rot[875:951,:], axis=1))

    # potential in 4S with db

    four_sphere = LFPy.FourSphereVolumeConductor(eeg_coords_4s, radii, sigmas)
    start_4s = time.time()
    eeg_4s = four_sphere.get_dipole_potential(P_rot, P_loc_4s)
    end_4s = time.time()
    time_4s = end_4s - start_4s
    # when comparing execution times:
    # eeg_4s = eeg_4s[:, 875:951]
    print('execution time 4S:', time_4s)
    # subtract DC-component
    dc_comp_4s = np.mean(eeg_4s[:, 0:25], axis=1)
    dc_comp_4s = np.expand_dims(dc_comp_4s, 1)
    eeg_4s -= dc_comp_4s
    eeg_4s *= 1e3  # from mV to uV

    # calculate EEGs with NYHeadModel
    start_nyh = time.time()
    nyh.calculate_eeg_signal(normal=True)
    end_nyh = time.time()
    time_nyh = end_nyh - start_nyh
    print('execution time NYH:', time_nyh)
    eeg_nyh = nyh.eeg[:, 875:951] # pV
    # subtract DC-component
    dc_comp_nyh = np.mean(eeg_nyh[:, 0:25], axis=1)
    dc_comp_nyh = np.expand_dims(dc_comp_nyh, 1)
    eeg_nyh -= dc_comp_nyh
     # from pV to uV
    eeg_nyh *= 1e-6
    elec_dists_nyh = (np.sqrt(np.sum((np.array(nyh.dipole_pos)[:, None] -
                                      np.array(nyh.elecs[:3, :]))**2, axis=0)))
    eeg_coords_nyh = nyh.elecs
    # some info
    # max_eeg = np.max(np.abs(eeg_nyh[:, time_idx]))
    # max_eeg_idx = np.argmax(np.abs(eeg_nyh[:, time_idx]))
    # max_eeg_pos = eeg_coords_nyh[:3, max_eeg_idx]

    dist, closest_elec_idx = nyh.find_closest_electrode()
    print("Closest electrode to dipole: {:1.2f} mm".format(dist))

    tvec = np.arange(P.shape[1]) + 875

    np.savez('../data/figure7_%s.npz' % dip_loc,
        radii = radii,
        p_rot = P_rot,
        p_loc_4s = P_loc_4s,
        p_loc_nyh = P_loc_nyh,
        eeg_coords_4s = eeg_coords_4s,
        eeg_coords_nyh = eeg_coords_nyh,
        elec_dists_4s = elec_dists_4s,
        elec_dists_nyh = elec_dists_nyh,
        eeg_4s = eeg_4s,
        eeg_nyh = eeg_nyh,
        time_idx = time_idx,
        tvec= tvec,
    )


if __name__ == '__main__':
    dip_loc = 'parietal_lobe'
    generate_eegs_nyh_n_4s(dip_loc)

    dip_loc = 'occipital_lobe'
    generate_eegs_nyh_n_4s(dip_loc)

