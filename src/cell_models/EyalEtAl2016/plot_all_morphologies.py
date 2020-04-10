import os
import sys
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
# import neuron
import LFPy

root_folder = "."
lab = "hbp_collab"

morphology_folder = join(root_folder, 'morphs')
morph_fig_folder = join(root_folder)
morph_list = [f for f in os.listdir(morphology_folder) if f.endswith("ASC")]

# for morph_name in morph_list[1:]:
idx = 0#int(sys.argv[1])
morph_name = morph_list[idx]
print(morph_name)
plt.close("all")
# morphology_name = '050217-zA.CNG.swc'
cell_parameters = {
        'morphology': join(morphology_folder, morph_name),
        'v_init': -70,
        'passive': True,
        'nsegs_method': "lambda_f",
        "lambda_f": 100,
        'dt': 2**-4,  # [ms] Should be a power of 2
        'tstart': 0,  # [ms] Simulation start time
        'tstop': 30,  # [ms] Simulation end time
        "pt3d": True,
}

cell = LFPy.Cell(**cell_parameters)

# print cell.totnsegs
cell.set_rotation(x=-np.pi/2)
# cell.set_rotation(z=np.pi/2)
# cell.set_rotation(y=np.pi/2)
# cell.set_rotation(x=np.pi/2, y=-0.0)

fig = plt.figure(figsize=[18, 9])
fig.subplots_adjust(wspace=0.2, top=0.97, bottom=0.04)
ax1 = fig.add_subplot(131, aspect=1, frameon=False, xlim=[-300, 300], ylim=[-500, 1500])
ax2 = fig.add_subplot(132, aspect=1, frameon=False, xlim=[-300, 300], ylim=[-500, 1500])
ax3 = fig.add_subplot(133, aspect=1, frameon=False, xlim=[-300, 300], ylim=[-300, 300])

from matplotlib.collections import PolyCollection


zips = []
for x, z in cell.get_idx_polygons():
    zips.append(list(zip(x, z)))
polycol = PolyCollection(zips,
                         edgecolors='none',
                         facecolors='k')
ax1.add_collection(polycol)

zips = []
for x, y in cell.get_idx_polygons(('x', 'y')):
    zips.append(list(zip(x, y)))
polycol = PolyCollection(zips,
                         edgecolors='none',
                         facecolors='k')
ax2.add_collection(polycol)

zips = []
for z, y in cell.get_idx_polygons(('z', 'y')):
    zips.append(list(zip(z, y)))
polycol = PolyCollection(zips,
                         edgecolors='none',
                         facecolors='k')
ax3.add_collection(polycol)


plt.savefig(join(morph_fig_folder, "%s.png" % morph_name))
