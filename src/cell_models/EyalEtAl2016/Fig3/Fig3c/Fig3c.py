########
#
# This code generate Fig 3c in Eyal ey al. 2016
# Here we examin the impact of Cm on the velocity of spike propagation along the axon
# 
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########

from neuron import h,gui
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
h.load_file("import3d.hoc")
h.load_file("nrngui.hoc")


h("objref cell, tobj")


# create the model:
morph_file = "../../morphs/2013_03_06_cell08_876_H41_05_Cell2.ASC"
model_path = "../../ActiveModels/"
model_name = 'model_0603_cell08_cm045'
h.load_file(model_path+model_name+".hoc")
h.execute("cell = new "+model_name+"()") 
nl = h.Import3d_Neurolucida3()
nl.quiet = 1
nl.input(morph_file)
imprt = h.Import3d_GUI(nl,0)
imprt.instantiate(h.cell)	
cell = h.cell
cell.geom_nseg()
cell.delete_axon()
cell.create_axon()
cell.biophys()
cell.active_biophys()


# Measure distances along the axon
L0 = cell.soma[0].L/2
L1 = cell.axon[0].L
L2 = cell.axon[1].L

# Compute the length_constant of the axon
RM = 1/cell.axon[0].g_pas
RA = cell.axon[0].Ra
d = cell.axon[0].diam/10**4
E_PAS = cell.soma[0].e_pas
length_constant = math.sqrt(RM/RA*d/4)
print "length_constant for this model:", length_constant
length_constant_um = length_constant*10000

# simulation parameters
h.steps_per_ms = 100
h.dt = 1/h.steps_per_ms 
h.tstop = 200
h.v_init = E_PAS


iclamp = h.IClamp(0.5,cell.soma[0])
iclamp.delay = 10
iclamp.dur = 1
iclamp.amp = 3


# Record voltage
v_soma = h.Vector()
v_axon_0 = h.Vector()
v_axon_1 = h.Vector()
tvec = h.Vector()
v_soma.record(cell.soma[0](0.5)._ref_v)
v_axon_0.record(cell.axon[0](1)._ref_v)
v_axon_1.record(cell.axon[1](1)._ref_v)
tvec.record(h._ref_t)


# init and run the NEURON simulation
def run_exp():

	h.init(h.v_init)
	h.run()

	np_v_soma = np.array(v_soma)	
	np_v_axon_0 = np.array(v_axon_0)	
	np_v_axon_1 = np.array(v_axon_1)	

	np_t = np.array(tvec)

	return np_t,np_v_soma,np_v_axon_0,np_v_axon_1	


print ""
print "spike initates in soma 0.5:", L0, " from axon initiation"
print "spike is measured ",L1, "um from soma which are ",L1/length_constant_um,"length_constants"
print "The axon continues ",L2, "from there which are ",L2/length_constant_um," length_constants"



np_t_cm045,np_v_soma_cm045,np_v_axon_0_cm045,np_v_axon_1_cm045 = run_exp()


time_diff1 = h.dt*(np.argmax(np_v_axon_0_cm045)-np.argmax(np_v_soma_cm045))
print ""
print "model with cm = 0.45"
print "peak time difference between soma and 1 mm into the axon: ", time_diff1
time_diff2 = h.dt*(np.argmax(np_v_axon_1_cm045)-np.argmax(np_v_soma_cm045))

print "peak time difference between soma and end of the axon: ", time_diff2
print "conductance velocity in the first 1 mm of the axon: ",((L1+L0/2)/10**6)/(time_diff1/10**3),"m/s"
print "average conductance conductance velocity in the entire axon: ",((L2+L1+L0/2)/10**6)/(time_diff2/10**3),"m/s"


cell.change_cm(2)
iclamp.amp = 4.2 # more input is needed in order to create the same spike in the larger cm case (See Fig 3b)

np_t_cm09,np_v_soma_cm09,np_v_axon_0_cm09,np_v_axon_1_cm09 = run_exp()



time_diff1 = h.dt*(np.argmax(np_v_axon_0_cm09)-np.argmax(np_v_soma_cm09))
print ""
print "model with cm = 0.9"
print "peak time difference between soma and 1 mm into the axon: ", time_diff1
time_diff2 = h.dt*(np.argmax(np_v_axon_1_cm09)-np.argmax(np_v_soma_cm09))

print "peak time difference between soma and end of the axon: ", time_diff2
print "conductance velocity in the first 1 mm of the axon: ",((L1+L0/2)/10**6)/(time_diff1/10**3),"m/s"
print "average conductance conductance velocity in the entire axon: ",((L2+L1+L0/2)/10**6)/(time_diff2/10**3),"m/s"

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)

ax.plot(np_t_cm045,np_v_soma_cm045,'--r',label='Cm = 0.45 somatic spike')
ax.plot(np_t_cm09,np_v_soma_cm09,'--b',label='Cm = 0.9  somatic spike')

ax.plot(np_t_cm045,np_v_axon_0_cm045,'r',label='Cm = 0.45 axonal spike')

ax.plot(np_t_cm09,np_v_axon_0_cm09,'b',label='Cm = 0.9  axonal spike')
plt.legend(loc='best', fancybox=True, framealpha=0.5)
plt.xlabel('Time (ms)',fontsize=14)
plt.ylabel('Voltage (mV)',fontsize=14)
plt.title('Eyal et al. Fig. 3c',fontsize = 16)

ax.axis([7,21,-90,40])
plt.tight_layout()
plt.show()





