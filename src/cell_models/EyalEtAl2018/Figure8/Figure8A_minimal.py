########
#
# This code generates a shorter version of figure 8A in Eyal et al 2017
# It simulates synchronous activation of synapses and the probability to spike as function of the number of activated synapses.
# Two cases are shown, the case of distributed activation and the case of clustered activation
# This is a short version, because in the paper much more cases and seeds per case where tested
#  
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########

# os.system('nrnivmodl ../ActiveMechanisms/')

import numpy as np
from neuron import h
import neuron
import matplotlib.pyplot as plt

h.load_file('stdrun.hoc')
h.load_file('stdlib.hoc')    #NEURON std. library
h.load_file('import3d.hoc')  #import 3D morphology lib


model = "cell0603_11_model_937"
h.load_file("import3d.hoc")
neuron.load_mechanisms("../ActiveMechanisms/")
h.load_file("../ActiveModels/"+model+".hoc")
h("objref HCell")
h("HCell = new "+model+"()")
HCell = h.HCell
nl = h.Import3d_Neurolucida3()

# Creating the model
nl.quiet = 1
nl.input("../Morphs/2013_03_06_cell11_1125_H41_06.ASC")
imprt = h.Import3d_GUI(nl, 0)   
imprt.instantiate(HCell)    
HCell.indexSections(imprt)
HCell.geom_nsec()   
HCell.geom_nseg()
HCell.delete_axon()
HCell.insertChannel()
HCell.init_biophys()
HCell.biophys()

h.dt = 0.1
h.celsius = 37
h.v_init = -86

# The stimulus
icl = h.IClamp(0.5, sec=HCell.soma[0])
icl.dur = 10
icl.delay = 120.33
icl.amp = 1

h.tstop = 400

Vsoma = h.Vector()
Vsoma.record(HCell.soma[0](.5)._ref_v)
tvec = h.Vector()
tvec.record(h._ref_t)

h.init(h.v_init)
h.run()
plt.plot(tvec, Vsoma)
plt.savefig('control_before.png')