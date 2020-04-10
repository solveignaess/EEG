########
#
# This code generates the left example in Figure 8C (clustered synapses) 
# for more details see the comments in Figure8A_short_version.py
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########


import os

os.system('nrnivmodl ../ActiveMechanisms/')

import numpy as np
from neuron import h, gui
import matplotlib.pyplot as plt
import math

NUMBER_OF_CLUSTERS = 6
seed = 12
MAX_JITTER = 0 #synchronous activation
cluster_type = "sync_clusters"

model = "cell0603_11_model_937"
h.load_file("import3d.hoc")

h.load_file("../ActiveModels/"+model+".hoc")
h("objref HCell")
h("HCell = new "+model+"()")
HCell = h.HCell
nl = h.Import3d_Neurolucida3()

# Creating the model
nl.quiet = 1
nl.input("../Morphs/2013_03_06_cell11_1125_H41_06.asc")
imprt = h.Import3d_GUI(nl, 0)   
imprt.instantiate(HCell)    
HCell.indexSections(imprt)
HCell.geom_nsec()   
HCell.geom_nseg()
HCell.delete_axon()
HCell.insertChannel()
HCell.init_biophys()
HCell.biophys()

class config_params():
    pass

config = config_params()

config.CLUSTER_TYPE = None
config.TAU_1_AMPA = 0.3
config.TAU_2_AMPA = 1.8
# properties of the model from figure 4:
config.TAU_1_NMDA = 8.019 
config.TAU_2_NMDA = 34.9884
config.N_NMDA = 0.28011
config.GAMMA_NMDA = 0.0765685
config.AMPA_W = 0.00073027
config.NMDA_W = 0.00131038
config.NMDA_W_BLOCKED = 0
config.E_SYN = 0

config.Spike_time =131.4
config.SPINE_HEAD_X = 1
config.CLUSTER_L = 20
config.CLUSTER_SIZE = 20




h.steps_per_ms = 25
h.dt = 1.0/h.steps_per_ms
h.celsius = 37
h.v_init = -86

Stim1 = h.NetStim()
Stim1.interval=10000 
Stim1.start= config.Spike_time
Stim1.noise=0
Stim1.number=1

config.stim = Stim1

h.tstop = config.Spike_time+150

from Figure8Functions import *

rd = h.Random(seed)

synaptic_locations = fill_clustered_synapses_demo(HCell,NUMBER_OF_CLUSTERS,rd,config)
SynList = []
ConList = []
add_synapses_on_list_of_segments(synaptic_locations,HCell,SynList,ConList,config)
configure_synaptic_delayes(MAX_JITTER,ConList,rd,config,cluster_type = cluster_type)


Vsoma = h.Vector()
Vsoma.record(HCell.soma[0](.5)._ref_v)
tvec = h.Vector()
tvec.record(h._ref_t)

h.init(h.v_init)
h.run()

np_t = np.array(tvec)
np_v =  np.array(Vsoma)

plt.plot(np_t,np_v,c='m')
plt.xlim(100,250)
plt.ylim(-100,50)

shp = h.Shape()
shp.rotate(0,0,0,0,0,3.4)

shp.show(0)
for i in range(NUMBER_OF_CLUSTERS*config.CLUSTER_SIZE):
    shp.point_mark(SynList[i], 6,4,6)

shp.view(-400,-600,800,1600,800,0,400,800)


ps = h.PlotShape() 
ps.exec_menu("View = plot")
ps.rotate(0,0,0,0,0,3.4)
ps.view(-400,-600,800,1600,800,0,400,800)
ps.variable("v")
h.load_file("../NEURON_color_maps/TColorMap.hoc")

cm1 = h.TColorMap("../NEURON_color_maps/jet.cm")


cm1.set_color_map(ps,-90,-10)
h.fast_flush_list.append(ps)
ps.exec_menu("Shape Plot")
ps.exec_menu("Show Diam")


ps.exec_menu("Variable Scale")

h.tstop = 151.5
h.run()


