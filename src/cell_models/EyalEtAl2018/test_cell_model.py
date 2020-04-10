
import os

# os.system('nrnivmodl ../ActiveMechanisms/')

import scipy.io as sio
import glob
import numpy as np
import neuron
from neuron import h
import matplotlib.pyplot as plt
import math


models_dirs = {}
models_dirs['cell0603_08'] = {'model': 'cell0603_08_model_602',
                              'morph': '2013_03_06_cell08_876_H41_05_Cell2.asc',
							  'stim_amp': 0.7,
                              'color': [0.0/255.0,161.0/255,75.0/255.0]}
cell = 'cell0603_08'


# h.load_file('stdlib.hoc')
# h.load_file("import3d.hoc")
# h.steps_per_ms = 100
h.dt = 1.0/100
# h.t_stop = 1500
h.celsius = 37
h.v_init = -86



model = models_dirs[cell]['model']
print("simulates model", model)
h.load_file(models_path+model+".hoc")
h("objref HCell")
h("HCell = new "+model+"()")
HCell = h.HCell
nl = h.Import3d_Neurolucida3()

# Creating the model
nl.quiet = 1
nl.input(morphs_path+models_dirs[cell]['morph'])
imprt = h.Import3d_GUI(nl, 0)
imprt.instantiate(HCell)
HCell.indexSections(imprt)
HCell.geom_nsec()
HCell.geom_nseg()
HCell.delete_axon()
HCell.insertChannel()
HCell.init_biophys()
HCell.biophys()

# The stimulus
icl = h.IClamp(0.5,sec=HCell.soma[0])
icl.dur = 1000
icl.delay = 120.33
amp = models_dirs[cell]['stim_amp']
icl.amp = amp

# Record the voltage at the soma
Vsoma = h.Vector()
Vsoma.record(HCell.soma[0](.5)._ref_v)
tvec = h.Vector()
tvec.record(h._ref_t)

HCell.soma[0].push()

h.init(h.v_init)
h.run()

np_t = np.array(tvec)
np_v = np.array(Vsoma)

models_dirs[cell]['t'] = np_t
models_dirs[cell]['v'] = np_v

if save_traces:
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.savetxt(save_path+model+"_sim_"+str(int(amp*1000))+'pA.txt',np.array([np_t,np_v]).T)
