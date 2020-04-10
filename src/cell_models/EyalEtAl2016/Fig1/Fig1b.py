########
#
# This code generate Fig. 1b in Eyal et al. 2016
# A close fit between model and experimental transients, with identical cable parameters as in Fig. 1a,
# was obtained for a range of (depolarizing and hyperpolarizing) voltage transients 
# recorded experimentally from that same cell
# 
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########

from neuron import h,gui
import numpy as np
import matplotlib.pyplot as plt
h.load_file("import3d.hoc")
h.load_file("nrngui.hoc")

# Create the model
h("objref cell, tobj")
morph_file = "../morphs/2013_03_06_cell08_876_H41_05_Cell2.ASC"
model_file = "model_0603_cell08"
model_path = "../PassiveModels/"
h.load_file(model_path+model_file+".hoc")
h.execute("cell = new "+model_file+"()") #replace?

nl = h.Import3d_Neurolucida3()
nl.quiet = 1
nl.input(morph_file)
imprt = h.Import3d_GUI(nl,0)
imprt.instantiate(h.cell)	
cell = h.cell
cell.geom_nseg()
cell.delete_axon()	
cell.biophys()


INJ = 27.06
DUR = 2
TSTOP = 129.06
INJ_AMP = 0.2
E_PAS = -86

BRIDGE_START = INJ-1 #We fitted the transient between 1 ms and 100 after the stimulus ended (same as  Major et al. 1993.)
BRIDGE_END = INJ + DUR+ 1

# load the experimental transients
data = []
data.append(np.loadtxt('Voltage_traces_1AB/p200pA_average_e86.dat',skiprows=2))
data.append(np.loadtxt('Voltage_traces_1AB/p100pA_average_e86.dat',skiprows=2))
data.append(np.loadtxt('Voltage_traces_1AB/p050pA_average_e86.dat',skiprows=2))
data.append(np.loadtxt('Voltage_traces_1AB/m050pA_average_e86.dat',skiprows=2))
data.append(np.loadtxt('Voltage_traces_1AB/m100pA_average_e86.dat',skiprows=2))
data.append(np.loadtxt('Voltage_traces_1AB/m200pA_average_e86.dat',skiprows=2))
plt.figure(1)

for (ix,d) in enumerate(data):
		
	v = d[:,1]
	if not ix:
		t = d[:,0]
		dt = t[1]-t[0]
		t1 = np.arange(0,BRIDGE_START+dt,dt)-INJ
		t2 = np.arange(BRIDGE_END,TSTOP+dt,dt)-INJ

	v1 = v[0:int(BRIDGE_START/dt)+1]
	v2 = v[int(BRIDGE_END/dt):v.size]

	plt.plot(t1,v1,linewidth=1,color='k')
	if ix==0:
		plt.plot(t2,v2,linewidth=1,color='k',label = 'exp')
	else:
		plt.plot(t2,v2,linewidth=1,color='k')


# simulation parameters
soma = cell.soma[0]
inj = h.IClamp(0.5, sec=soma)
inj.dur = DUR
inj.delay = INJ 
h.v_init = E_PAS
h.tstop = TSTOP


Vvec = h.Vector()
Vvec.record(soma(0.5)._ref_v)

# Run and record the model for each of the short current inputs
for ix,amp in enumerate([-0.2,-0.1,-0.05,0.05,0.1,0.2]):
	inj.amp = amp
	h.init(h.v_init)
	h.run()
	v = np.array(Vvec)
	v1 = v[0:int(BRIDGE_START/h.dt)+1]
	v2 = v[int(BRIDGE_END/h.dt)+1:v.size]
	if not ix:
		t1 = np.arange(0,BRIDGE_START,h.dt)-INJ
		t2 = np.arange(BRIDGE_END,TSTOP,h.dt)-INJ


	plt.plot(t1,v1,linewidth=3,color='g')
	if ix ==0:
		plt.plot(t2,v2,linewidth=3,color='g',label = 'model')
	else:
		plt.plot(t2,v2,linewidth=3,color='g')

plt.axis([-10,75,-87.5,-84.5])

plt.legend(loc='best', fancybox=True, framealpha=0.5)
plt.xlabel('Time (ms)',fontsize = 14)
plt.ylabel('Voltage (mV)',fontsize = 14)
plt.title('Eyal et al. Fig. 1b',fontsize = 16)

plt.show()


