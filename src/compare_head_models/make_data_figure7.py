from os.path import join
import numpy as np
import LFPy
import time

from equidistal_surface_points import return_equidistal_xyz


def generate_eegs_nyh_n_4s(dip_loc):

    # get population dipole from hybrid simulation:
    cdm_file = join('..', "hybrid_EEG_evoked", "evoked_cdm", "summed_cdm.npy")
    # Only interested in time around evoked potential (875 - 951 ms)
    P = np.load(cdm_file).T[:, 875:951]

    # Load New York Head model
    nyh = LFPy.NYHeadModel()
    # place dipole and find normal vector
    P_loc_nyh = nyh.dipole_pos_dict[dip_loc]
    nyh.set_dipole_pos(dipole_pos=P_loc_nyh)
    P_loc_idx = nyh.return_closest_idx(P_loc_nyh)
    norm_vec = nyh.cortex_normals[:, P_loc_idx]

    # set four-sphere properties
    # From Huang et al. (2013): 10.1088/1741-2560/10/6/066004
    sigmas = [0.276, 1.65, 0.01, 0.465]
    radii = [89000., 90000., 95000., 100000.]
    rad_tol = 1e-2
    x_eeg, y_eeg, z_eeg = return_equidistal_xyz(224, r=radii[-1] - rad_tol)
    eeg_coords_4s = np.array([x_eeg, y_eeg, z_eeg]).T
    # Place the dipole 1000 µm down into cortex
    dipole_radial_pos = radii[0] - 1000

    # find location on 4S
    # want to find position on brain, such that the location vector is
    # parallel to norm_vec from nyh.
    dipole_loc_4s = norm_vec*dipole_radial_pos
    # Rotate dipole to be aligned with cortex normal
    P_rot = nyh.rotate_dipole_to_surface_normal(P)

    elec_dists_4s = np.sqrt(np.sum((eeg_coords_4s - dipole_loc_4s)**2, axis=1))
    elec_dists_4s *= 1e-3 # convert from um to mm
    min_dist_idx = np.argmin(elec_dists_4s)
    print("Minimum 4o distance {:2.02f} mm".format(elec_dists_4s[min_dist_idx]))

    time_idx = np.argmax(np.linalg.norm(P_rot, axis=0))

    four_sphere = LFPy.FourSphereVolumeConductor(eeg_coords_4s, radii, sigmas)
    start_4s = time.time()
    eeg_4s = four_sphere.get_dipole_potential(P_rot, dipole_loc_4s)
    end_4s = time.time()
    time_4s = end_4s - start_4s
    print('execution time 4S:', time_4s)
    # subtract DC-component
    dc_comp_4s = np.mean(eeg_4s[:, 0:25], axis=1)
    dc_comp_4s = np.expand_dims(dc_comp_4s, 1)
    eeg_4s -= dc_comp_4s
    eeg_4s *= 1e3  # from mV to uV

    # calculate EEGs with NYHeadModel
    start_nyh = time.time()
    eeg_nyh = nyh.get_transformation_matrix() @ P_rot # pV
    end_nyh = time.time()
    time_nyh = end_nyh - start_nyh
    print('execution time NYH:', time_nyh)
    # subtract DC-component
    dc_comp_nyh = np.mean(eeg_nyh[:, 0:25], axis=1)
    dc_comp_nyh = np.expand_dims(dc_comp_nyh, 1)
    eeg_nyh -= dc_comp_nyh
     # from mV to uV
    eeg_nyh *= 1e3
    elec_dists_nyh = (np.sqrt(np.sum((np.array(nyh.dipole_pos)[:, None] -
                                      np.array(nyh.elecs[:3, :]))**2, axis=0)))
    eeg_coords_nyh = nyh.elecs

    dist, closest_elec_idx = nyh.find_closest_electrode()
    print("Closest electrode to dipole: {:1.2f} mm".format(dist))

    tvec = np.arange(P.shape[1]) + 875

    np.savez('../data/figure7_%s.npz' % dip_loc,
        radii = radii,
        p_rot = P_rot,
        p_loc_4s = dipole_loc_4s,
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

