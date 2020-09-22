# load data from file and plot eegs from population current dipole moment
# (hybrid) placed in visual cortex and on the right side of the head
import matplotlib
matplotlib.use("AGG")
from os.path import join
import numpy as np
import matplotlib.pyplot as plt


# load data from dipole placed in visual cortex
data_back_of_head = np.load('../data/figure7_occipital_lobe.npz')
radii = data_back_of_head['radii']
P_rot_back = data_back_of_head['p_rot']
P_loc_4s_back = data_back_of_head['p_loc_4s']
P_loc_nyh_back = data_back_of_head['p_loc_nyh']
eeg_coords_4s = data_back_of_head['eeg_coords_4s']
eeg_coords_nyh = data_back_of_head['eeg_coords_nyh'] * 1000
elec_dists_4s_back = data_back_of_head['elec_dists_4s']
elec_dists_nyh_back = data_back_of_head['elec_dists_nyh']
eeg_4s_back = data_back_of_head['eeg_4s']
eeg_nyh_back = data_back_of_head['eeg_nyh']
time_idx = data_back_of_head['time_idx']
tvec = data_back_of_head['tvec']

# load data from dipole placed on the side of cortex
data_side_top_sulcus = np.load('../data/figure7_parietal_lobe.npz')

P_rot_side = data_side_top_sulcus['p_rot']
P_loc_4s_side = data_side_top_sulcus['p_loc_4s']
P_loc_nyh_side = data_side_top_sulcus['p_loc_nyh']
elec_dists_4s_side = data_side_top_sulcus['elec_dists_4s']
elec_dists_nyh_side = data_side_top_sulcus['elec_dists_nyh']
eeg_4s_side = data_side_top_sulcus['eeg_4s']
eeg_nyh_side = data_side_top_sulcus['eeg_nyh']


def plot_coord_syst():
    ax_xy_2 = fig.add_axes([0.185, 0.42, 0.05, 0.05])
    ax_xy_3 = fig.add_axes([0.35, 0.41, 0.05, 0.05])
    ax_xz_6 = fig.add_axes([0.17, 0.13, 0.05, 0.05])
    ax_xz_7 = fig.add_axes([0.33, 0.13, 0.05, 0.05])
    for ax in [ax_xy_2, ax_xy_3, ax_xz_6, ax_xz_7]:
        ax.arrow(0, 0, 12, 0, head_width=5, head_length=5, fc='k', ec='k', lw=2)
        ax.arrow(0, 0, 0, 12, head_width=5, head_length=5, fc='k', ec='k', lw=2)
        ax.set_xlim([-6, 21])
        ax.set_ylim([-6, 21])
        ax.set_aspect('equal')
        ax.axis('off')
    for ax in [ax_xy_2, ax_xy_3]:
        ax.text(20, -3, 'x', fontsize=10)
        ax.text(-2, 23, 'y', fontsize=10)
    for ax in [ax_xz_6, ax_xz_7]:
        ax.text(20, -3, 'x', fontsize=10)
        ax.text(-2, 23, 'z', fontsize=10)

################################################################################
################################# make figure ##################################
################################################################################

plt.close('all')
fig = plt.figure()
fig.set_size_inches(9, 5.5)

xstart = 0.16
xwidth = 0.15
ystart_1 = 0.72
ystart_2 = 0.47
ystart_3 = 0.15
ywidth = 0.26
ax_cort_o = fig.add_axes([0.025, 0.07, 0.2, 0.3], aspect=1, xticks=[],
                         yticks=[], frameon=False)
ax_cort_p = fig.add_axes([0.025, 0.4, 0.2, 0.3], aspect=1, xticks=[],
                         yticks=[], frameon=False)

ax_4s = fig.add_axes([xstart + 0.164, 0.73, 0.18, 0.22], aspect=1,
                     title='four-sphere', frameon=False, xticks=[],
                     yticks=[], xlim=(-110000, 110000)) # 4S model
ax_NYH = fig.add_axes([xstart + 0.02, 0.73, 0.15, 0.22], aspect=1,
                      title='New York head', frameon=False, xticks=[], yticks=[]) # NYH model
ax2 = fig.add_axes([xstart+0.18, ystart_2-0.015, xwidth, ywidth],
                   frameon=False, xticks=[], yticks=[],
                   xlim=(-110000, 110000), ylim=(-110000, 110000)) # electrodes parietal 4S
ax6 = fig.add_axes([xstart+0.18, ystart_3-0.015, xwidth, ywidth],
                   frameon=False, xticks=[], yticks=[],
                   xlim=(-110000, 110000), ylim=(-110000, 110000)) # electrodes occipital 4S
ax3 = fig.add_axes([xstart+0.02, ystart_2, xwidth, ywidth],
                   frameon=False, xticks=[], yticks=[],
                   xlim=(-110000, 110000), ylim=(-110000, 110000))  # electrodes parietal NYH
ax7 = fig.add_axes([xstart+0.02, ystart_3, xwidth, ywidth],
                   frameon=False, xticks=[], yticks=[],
                   xlim=(-110000, 110000), ylim=(-110000, 110000)) # electrodes occipital NYH
ax4 = fig.add_axes([xstart+0.41, ystart_2, 0.15, ywidth],
                   ylabel=r'$\Phi$ ($\mu$V)') # closest EEG parietal
ax8 = fig.add_axes([xstart+0.41, ystart_3, 0.15, ywidth],
                   xlabel="t (ms)", ylabel=r'$\Phi$ ($\mu$V)') # closest EEG occipital
ax5 = fig.add_axes([xstart+0.66, ystart_2, 0.15, ywidth],
                   ylabel='|$\Phi$($t = t_{max}$)|  ($\mu$V)') # EEG at electrodes parietal
ax9 = fig.add_axes([xstart+0.66, ystart_3, 0.15, ywidth],
                   xlabel='electrode distance (mm)',
                   ylabel='|$\Phi$($t = t_{max}$)| ($\mu$V)') # EEG at electrodes occipital

#ax2.set_title(r'4S EEG($t = t_{max}$)', fontsize=10, pad=11)
#ax3.set_title(r'NYH EEG($t = t_{max}$)', fontsize=10, pad=11)
ax4.set_title('EEG at closest electrode', fontsize=10)


for ax in [ax3, ax7]:
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])


# show 4S-illustration
head_colors = plt.cm.Pastel1([0, 1, 2, 3])
radii_tweaked = [radii[0]] + [r + 500 for r in radii[1:]]
# plot 4s-model
for i in range(4):
    ax_4s.add_patch(plt.Circle((0, 0), radius=radii_tweaked[-1 - i],
                               color=head_colors[-1-i], fill=True, ec='k', lw=.1))

ax_4s.plot(P_loc_4s_back[0], P_loc_4s_back[2], 'ko', ms=0.00001)


# insert NYH illustration
img = plt.imread(join('head_model_illustrations', 'head_snapshot.png'))
ax_NYH.imshow(img, aspect='equal')

img = plt.imread(join('head_model_illustrations', 'cortex_parietal_lobe.png'))
ax_cort_p.imshow(img, aspect='equal')

img = plt.imread(join('head_model_illustrations', 'cortex_occipital_lobe.png'))
ax_cort_o.imshow(img, aspect='equal')

fig.text(0.005, 0.15, "dipole in\noccipital lobe", rotation=90, fontsize=15,)
fig.text(0.005, 0.48, "dipole in\nparietal lobe", rotation=90, fontsize=15)

x_eeg = eeg_coords_4s[:,0]
y_eeg = eeg_coords_4s[:,1]
z_eeg = eeg_coords_4s[:,2]

# indices to plot
backhead_idxs = np.where(np.array(y_eeg) < 0)[0]
upper_idxs = np.where(np.array(z_eeg) > 0)[0]
# colors for plotting
vmax = 0.2 #np.max(np.abs(eeg_4s_back[backhead_idxs, time_idx])) # 0.1
vmin = -vmax
clr = lambda phi: plt.cm.bwr((phi - vmin) / (vmax - vmin))

# eeg_elec_ms = 40
# plot 4S EEG electrodes dipole back of head

for idx in upper_idxs:
    ax2.plot(x_eeg[idx], y_eeg[idx], 'o', ms=8, mec='k', zorder=z_eeg[idx],
                c=clr(eeg_4s_side[idx, time_idx]))

ax2.plot(P_loc_4s_side[0], P_loc_4s_side[1], '*', ms=8, color='orange', zorder=1e6)

ax6.yaxis.set_label_coords(-.21, .44)

for idx in backhead_idxs:
    zorder = y_eeg[idx]
    ax6.plot(x_eeg[idx], z_eeg[idx], 'o',
            c=clr(eeg_4s_back[idx, time_idx]), ms=8, mec='k', zorder=-zorder, clip_on=False, mew=0.1)
ax6.plot(P_loc_4s_back[0], P_loc_4s_back[2], '*', ms=7, color='orange', zorder=1e6)

# fig.text(0.08, 0.7, 'max amplitude at EEG electrodes', fontsize=10)

# plot NYH EEG back of head:
# max_elec_idx = np.argmax(np.std(eeg_nyh_back, axis=1))

for idx in range(len(eeg_nyh_back)):
    ax3.plot(eeg_coords_nyh[0, idx], eeg_coords_nyh[1, idx], 'o', ms=8,
                    c=clr(eeg_nyh_side[idx, time_idx]), mec='k',
             zorder=eeg_coords_nyh[2, idx], clip_on=False)
ax3.plot(P_loc_nyh_side[0] * 1000, P_loc_nyh_side[1] * 1000, '*', ms=7, color='orange', zorder=1e6)

# plot EEG from NYH on head:
# max_elec_idx = np.argmax(np.std(eeg_nyh_side, axis=1))


for idx in range(len(eeg_nyh_side)):
    ax7.plot(eeg_coords_nyh[0, idx], eeg_coords_nyh[2, idx], 'o', ms=8, clip_on=False,
             c=clr(eeg_nyh_back[idx, time_idx]), mec='k', zorder=-eeg_coords_nyh[1, idx])

ax7.plot(P_loc_nyh_back[0] * 1000, P_loc_nyh_back[2] * 1000, '*', ms=7, color='orange', zorder=1e6)
plot_coord_syst()


# colorbar
cax = fig.add_axes([0.18, 0.075, 0.33, 0.01])
m = plt.cm.ScalarMappable(cmap=plt.cm.bwr)
ticks = np.linspace(vmin, vmax, 5) # global normalization
m.set_array(ticks)
cbar = fig.colorbar(m, cax=cax,
                    extend='both', orientation='horizontal')
cbar.outline.set_visible(False)
cbar.set_ticks(ticks)
plt.xticks(ticks, [str(round(tick,2)) for tick in ticks], fontsize=10)
cbar.set_label(r'$\Phi$ ($\rm \mu$V)', labelpad=-1, fontsize=10)

# plot EEG at closest electrode for 4S and NYH
# dipole in back of head
ax4.axvline(tvec[time_idx], ls=':', color='gray', lw=1)
l1, = ax4.plot(tvec, eeg_4s_side[np.argmin(elec_dists_4s_side), :], 'k', lw=2.5)
l2, = ax4.plot(tvec, eeg_nyh_side[np.argmin(elec_dists_nyh_side), :], 'gray', lw=2.5)

fig.legend([l2, l1], ["New York head", "four-sphere"], frameon=False, ncol=2, loc=(0.60, 0.8))

ax4.legend(fontsize=10, frameon=False, loc=(0.45, 0))


# dipole in side of head
ax8.axvline(tvec[time_idx], ls=':', color='gray', lw=1)
ax8.plot(tvec, eeg_4s_back[np.argmin(elec_dists_4s_back), :], 'k', lw=2.5,)
ax8.plot(tvec, eeg_nyh_back[np.argmin(elec_dists_nyh_back), :], 'gray', lw=2.5,)


# print RE between 4S and NYH for the two dip locs, using NYH as gold standard:
eeg_4s_side_t = eeg_4s_side[np.argmin(elec_dists_4s_side), time_idx]
eeg_nyh_side_t = eeg_nyh_side[np.argmin(elec_dists_nyh_side), time_idx]
re_side_t = np.abs(eeg_4s_side_t - eeg_nyh_side_t)/np.abs(eeg_nyh_side_t)
print('RE between 4S and NYH at tmax for dipole in parietal lobe:', re_side_t)

eeg_4s_back_t = eeg_4s_back[np.argmin(elec_dists_4s_back), time_idx]
eeg_nyh_back_t = eeg_nyh_back[np.argmin(elec_dists_nyh_back), time_idx]
re_back_t = np.abs(eeg_4s_back_t - eeg_nyh_back_t)/np.abs(eeg_nyh_back_t)
print('RE between 4S and NYH at tmax for dipole in occipital lobe:', re_back_t)

# formatting ax4 and ax8
x_ticks = [880, 900, 920, 940, 960] #+ [tvec[time_idx]]
x_ticklabels = [str(tick) for tick in x_ticks]
x_ticklabels[-1] = ' ' #r'$t_{max}$'
x_ticklabels[1] += '  '
for ax in [ax4, ax8]:
    ax.set_xlim([875, 950])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.set_xlim([875, 950])
    ax.set_ylim([-0.3, 0.1])

    # ax4.set_yticks([-0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.text(905, -0.355, r'$t_{max}$', fontsize=6)

# plot max EEG at electrode with increasing distance from top: 4S and NYH
# dipole back
ax5.scatter(elec_dists_nyh_side, np.max(np.abs(eeg_nyh_side), axis=1), s=14,
            c='gray',
            clip_on=False,)
ax5.scatter(elec_dists_4s_side, np.max(np.abs(eeg_4s_side), axis=1), s=14,
            c='k', clip_on=False,)

# dipole side
ax9.scatter(elec_dists_nyh_back, np.max(np.abs(eeg_nyh_back), axis=1), s=14,
            c='gray', clip_on=False,)
ax9.scatter(elec_dists_4s_back, np.max(np.abs(eeg_4s_back), axis=1), s=14,
            c='k', clip_on=False)


# format ax5, ax9
for ax in [ax5, ax9]:
    ax.set_ylim([vmin, vmax])
    # ax.set_ylabel(r'$|\Phi|_{max}$ ($\mu$V)', labelpad=7)
    ax.set_xlim([0, 230])
    ax.set_xticks([0., 100., 200.])
    ax.set_ylim([0, 0.25])
    ax.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# label axes
xpos = [0.20, 0.35, 0.07, 0.20, 0.35, 0.52, 0.75, 0.07, 0.20, 0.35, 0.52, .75]
ypos = [0.94, 0.94, 0.73, 0.73, 0.73, 0.73, 0.73, 0.4, 0.4, 0.4, 0.4, 0.4]
letters = 'ABCDEFGHIJKL'
for i in range(len(letters)):
    fig.text(xpos[i], ypos[i], letters[i],
         horizontalalignment='center',
         verticalalignment='center',
         fontweight='demibold',
         fontsize=12)

plt.savefig('../figures/Figure7.pdf', dpi=300)
