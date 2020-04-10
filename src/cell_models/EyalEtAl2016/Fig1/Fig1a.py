########
#
# This code generate Fig. 1a in Eyal et al. 2016
# The different colors present different attempts to fit the voltage transient with different values of Cm.
# The green case (Cm = 0.45) best fitted the experimental transient/
# Note that in all cases the values of Rm and Ra are the values that result from the praxis optimisation algorithm (see MEthods in Eyal et al.)
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


# display morphology in NEURON shape plot
shp=h.Shape()
shp.view(-300,-300,600,1200,200,0,400,800)
shp.rotate(0,0,0,0,0,2.8)

# brief depolarizing current pulse - same parameters to the experminet. 
INJ = 27.06
DUR = 2
TSTOP = 129.06
INJ_AMP = 0.2
E_PAS = -86

BRIDGE_START = INJ-1 #We fitted the transient between 1 ms and 100 after the stimulus ended (same as  Major et al. 1993.)
BRIDGE_END = INJ + DUR+ 1
COLOR_ARR = [[1,0,0],[1,0,1],[0,1,0]]

# load the experimental transient
data = np.loadtxt('Voltage_traces_1AB/p200pA_average_e86.dat',skiprows=2)
t_d = data[:,0]
v_d = data[:,1]
dt = t_d[1]-t_d[0]
t_1 = np.arange(0,BRIDGE_START+dt,dt)-INJ
t_2 = np.arange(BRIDGE_END,TSTOP+dt,dt)-INJ
v1_d = v_d[0:int(BRIDGE_START/dt)+1]
v2_d = v_d[int(BRIDGE_END/dt):v_d.size]

plt.figure(1)
plt.plot(t_1,v1_d,linewidth=2,color='k')
plt.plot(t_2,v2_d,linewidth=2,color='k',label='exp')

soma = cell.soma[0]
inj = h.IClamp(0.5, sec=soma)
inj.amp  = INJ_AMP
inj.dur = DUR
inj.delay = INJ 
h.v_init = E_PAS
h.tstop = TSTOP



# A function that changes the cable parameters of the model 
# We assumed that there are no spines in the most proximal dendrites (distance to soma<60)
def change_model_pas(CM,RM,RA,StepDist,F_Spines):
	for sec in cell.all:
		sec.cm = CM
		sec.g_pas = 1.0/RM
		sec.Ra = RA

	h.distance(0,sec=soma)

	for sec in h.cell.basal:
		for seg in sec:
			if h.distance(seg.x)>StepDist:
				seg.cm = CM*F_Spines
				seg.g_pas = (1.0/RM)*F_Spines


	for sec in h.cell.apical:
		for seg in sec:
			if h.distance(seg.x)>StepDist:
				seg.cm = CM*F_Spines
				seg.g_pas = (1.0/RM)*F_Spines


# Best model assuming Cm=1
change_model_pas(1,28981,629,60,1.9)

Vvec = h.Vector()
Vvec.record(soma(0.5)._ref_v)
h.run()
np_arr_cm1 = np.array(Vvec)

t_m1 = np.arange(0,BRIDGE_START,h.dt)-INJ
t_m2 = np.arange(BRIDGE_END,TSTOP,h.dt)-INJ
v1 = np_arr_cm1[0:int(BRIDGE_START/h.dt)+1]
v2 = np_arr_cm1[int(BRIDGE_END/h.dt)+1:v_d.size]
plt.plot(t_m1,v1,linewidth=2,color='r')
plt.plot(t_m2,v2,linewidth=2,color='r',label='Cm = 1')

# Best model assuming Cm=0.6

change_model_pas(0.6,35115,384,60,1.9)

Vvec = h.Vector()
Vvec.record(soma(0.5)._ref_v)
h.init(h.v_init)
h.run()
np_arr_cm0_6 = np.array(Vvec)
v1 = np_arr_cm0_6[0:int(BRIDGE_START/h.dt)+1]
v2 = np_arr_cm0_6[int(BRIDGE_END/h.dt)+1:v_d.size]
plt.plot(t_m1,v1,linewidth=2,color='m')
plt.plot(t_m2,v2,linewidth=2,color='m',label='Cm = 0.6')

# Best model (All three parameters were free in the optimisation)

change_model_pas(0.45234,38907,203.23,60,1.9) 

Vvec = h.Vector()
Vvec.record(soma(0.5)._ref_v)
h.init(h.v_init)
h.run()
np_arr_cm0_45 = np.array(Vvec)

v1 = np_arr_cm0_45[0:int(BRIDGE_START/h.dt)+1]
v2 = np_arr_cm0_45[int(BRIDGE_END/h.dt)+1:v_d.size]
plt.plot(t_m1,v1,linestyle='--',linewidth=3,color='g')
plt.plot(t_m2,v2,linestyle='--',linewidth=3,color='g',label='Cm = 0.45')

plt.axis([-10,40,-86.15,-84.55])

plt.legend(loc='best', fancybox=True, framealpha=0.5)
plt.xlabel('Time (ms)',fontsize = 14)
plt.ylabel('Voltage (mV)',fontsize = 14)
plt.title('Eyal et al. Fig. 1a',fontsize = 16)

plt.show()


