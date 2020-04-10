import os
import sys
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import neuron
from neuron import h
import LFPy

root_folder = "."

morphology_folder = join(root_folder, 'morphs')
morph_fig_folder = join(root_folder)
morph_list = [f for f in os.listdir(morphology_folder) if f.endswith("ASC")]

mod_folder = 'mechansims'
cell_model = "model_0603_cell08_cm045"
model_folder = "ActiveModels"
# for morph_name in morph_list[1:]:
idx = 0#int(sys.argv[1])
morph_name = "2013_03_06_cell08_876_H41_05_Cell2.ASC"
# print(morph_name)
plt.close("all")

neuron.load_mechanisms(mod_folder)
# morphology_name = '050217-zA.CNG.swc'
cell_parameters = {
        'morphology': join(morphology_folder, morph_name),
        'templatefile': join(model_folder, cell_model + '.hoc'),
        'templatename': cell_model,
        'templateargs': join(morphology_folder, morph_name),
        'nsegs_method': None,
        'v_init': -85,
        'passive': False,
        'dt': 2**-4,  # [ms] Should be a power of 2
        'tstart': -150,  # [ms] Simulation start time
        'tstop': 100,  # [ms] Simulation end time
        "pt3d": True,
}

cell = LFPy.TemplateCell(**cell_parameters)

cell.set_rotation(x=-np.pi/2)
# cell.set_rotation(z=np.pi/2)
cell.set_rotation(y=np.pi/8)
# cell.set_rotation(x=np.pi/2, y=-0.0)


print(cell.totnsegs)
# for seg in h.allsec():
#     print(seg.cm)
# Define synapse parameters
synapse_parameters = {
    'idx' : cell.get_closest_idx(x=0., y=0., z=0.),
    'e' : 0.,                   # reversal potential
    'syntype' : 'ExpSyn',       # synapse type
    'tau' : 5.,                 # synaptic time constant
    'weight' : .04,            # synaptic weight
    'record_current' : True,    # record synapse current
}

# Create synapse and set time of synaptic input
synapse = LFPy.Synapse(cell, **synapse_parameters)
synapse.set_spike_times(np.array([5.]))




cell.simulate(rec_vmem=True)

fig = plt.figure(figsize=[18, 9])
fig.subplots_adjust(wspace=0.2, top=0.97, bottom=0.04)
ax1 = fig.add_subplot(131, aspect=1, frameon=False, xlim=[-300, 300], ylim=[-500, 1500])
ax2 = fig.add_subplot(132,)
# ax3 = fig.add_subplot(133, aspect=1, frameon=False, xlim=[-300, 300], ylim=[-300, 300])

from matplotlib.collections import PolyCollection


zips = []
for x, z in cell.get_idx_polygons():
    zips.append(list(zip(x, z)))
polycol = PolyCollection(zips,
                         edgecolors='none',
                         facecolors='k')
ax1.add_collection(polycol)

ax2.plot(cell.tvec, cell.vmem[cell.somaidx[0], :])

plt.savefig(join(morph_fig_folder, "cell_model_%s.png" % morph_name))
