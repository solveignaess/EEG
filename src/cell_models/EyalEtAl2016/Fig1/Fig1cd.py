########
#
# This code generate Fig. 1d in Eyal et al. 2016
# We optimised the cable parameters for five other human HL2/3 pyramidal cells (Fig. 1c1-c5)
# the low value for Cm, arounf 0.5 uF/cm2, was found in all these five additional modeled cells.
#
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

# Stimulation parameters
INJ = 27.06
DUR = 2
BRIDGE_START = INJ-1 
BRIDGE_END = INJ + DUR+ 1
TSTOP = 129.06
INJ_AMP = 0.2
E_PAS = -86
h.dt = 0.025


# The current to this cell was injected in different timing (but other than that exactly the same)
INJ_0306_cell03 = 25.5
BRIDGE_START_0306_cell03 = INJ_0306_cell03-1
BRIDGE_END_0306_cell03 = INJ_0306_cell03+DUR+2
TSTOP_0306_cell03 = 127.5

# A function that create neuron model based on template model_name and morph_file 
def create_model(model_name,morph_file):
	h("objref cell, tobj")
	morph_file = "../morphs/"+morph_file
	model_path = "../PassiveModels/"
	h.load_file(model_path+model_name+".hoc")
	h.execute("cell = new "+model_name+"()") #replace?
	nl = h.Import3d_Neurolucida3()
	nl.quiet = 1
	nl.input(morph_file)
	imprt = h.Import3d_GUI(nl,0)
	imprt.instantiate(h.cell)	
	cell = h.cell
	cell.geom_nseg()
	cell.delete_axon()	
	cell.biophys()

	return cell

# Run the experiment: inject the short current injection the model's soma and record the voltage
def run_exp(cell,delay,tstop):
	soma = cell.soma[0]
	inj = h.IClamp(0.5, sec=soma)
	inj.amp  = INJ_AMP
	inj.dur = DUR
	inj.delay = delay 
	h.v_init = E_PAS
	h.tstop = tstop
	Vvec = h.Vector()
	Vvec.record(soma(0.5)._ref_v)
	Tvec = h.Vector()
	Tvec.record(h._ref_t)
	h.init(h.v_init)
	h.run()
	T = np.array(Tvec)-delay
	V = np.array(Vvec)
	return T,V



# data files - each is a voltage transient resulting from averaging 50 transients 
files = ["0313_cell03_200pA_average_e86","","0306_cell11_200pA_average_e86",
			"0313_cell06_200pA_average_e86","0313_cell05_200pA_average_e86"]

plt.figure(1,figsize=(16,5))
plt.title('Eyal et al. Fig. 1d',fontsize = 16)

# plot the voltage transients
for ix,f in enumerate(files):
	if not f:
		continue
	data = np.loadtxt('Voltage_traces_1CD/'+f+'.dat',skiprows=2)
	v = data[:,1]
	if not ix:
		t = data[:,0]-INJ
		dt = t[1]-t[0]
		t1 = t[0:int(BRIDGE_START/dt)+1]
		t2 = t[int(BRIDGE_END/dt):v.size]

	v1 = v[0:int(BRIDGE_START/dt)+1]
	v2 = v[int(BRIDGE_END/dt):v.size]

	plt.subplot(1,5,ix+1)
	plt.plot(t1,v1,linewidth=1,color='k')
	plt.plot(t2,v2,linewidth=1,color='k',label = 'exp')

f_0306_cell03 = '0306_cell03_200pA_average_e86'

data = np.loadtxt('Voltage_traces_1CD/'+f_0306_cell03+'.dat',skiprows=2)
v = data[:,1]
t = data[:,0]-INJ_0306_cell03
dt = t[1]-t[0]
t1 = t[0:int(BRIDGE_START_0306_cell03/dt)+1]
t2 = t[int(BRIDGE_END_0306_cell03/dt):v.size]

v1 = v[0:int(BRIDGE_START_0306_cell03/dt)+1]
v2 = v[int(BRIDGE_END_0306_cell03/dt):v.size]
plt.subplot(1,5,2)
plt.plot(t1,v1,linewidth=1,color='k')
plt.plot(t2,v2,linewidth=1,color='k',label = 'exp')



# Create the models and plot their response
plt.subplot(1,5,1)

cell_1303_cell03 = create_model("model_1303_cell03","2013_03_13_cell03_1204_H42_02.ASC")

T,V = run_exp(cell_1303_cell03,INJ,TSTOP)
plt.plot(T[0:int(BRIDGE_START/h.dt)+1],V[0:int(BRIDGE_START/h.dt)+1],linestyle='--',
												LineWidth=4,color=[248.0/255,153.0/255,55.0/255])
plt.plot(T[int(BRIDGE_END/h.dt)+1:v.size],V[int(BRIDGE_END/h.dt)+1:v.size],linestyle='--',
												LineWidth=4,color=[248.0/255,153.0/255,55.0/255],label = 'model')

plt.subplot(1,5,2)

model_0603_cell03 = create_model("model_0603_cell03","2013_03_06_cell03_789_H41_03.ASC")
T,V = run_exp(model_0603_cell03,INJ_0306_cell03,TSTOP_0306_cell03)
plt.plot(T[0:int(BRIDGE_START_0306_cell03/h.dt)+1],V[0:int(BRIDGE_START_0306_cell03/h.dt)+1],linestyle='--',
												LineWidth=4,color=[98.0/255,166.0/255,219.0/255])
plt.plot(T[int(BRIDGE_END_0306_cell03/h.dt)+1:v.size],V[int(BRIDGE_END_0306_cell03/h.dt)+1:v.size],linestyle='--',
												LineWidth=4,color=[98.0/255,166.0/255,219.0/255],label = 'model')


plt.subplot(1,5,3)

cell_0603_cell11 = create_model("model_0603_cell11","2013_03_06_cell11_1125_H41_06.ASC")
T,V = run_exp(cell_0603_cell11,INJ,TSTOP)
plt.plot(T[0:int(BRIDGE_START/h.dt)+1],V[0:int(BRIDGE_START/h.dt)+1],linestyle='--',
												LineWidth=4,color=[237.0/255,31.0/255,36.0/255])
plt.plot(T[int(BRIDGE_END/h.dt)+1:v.size],V[int(BRIDGE_END/h.dt)+1:v.size],linestyle='--',
												LineWidth=4,color=[237.0/255,31.0/255,36.0/255],label = 'model')

plt.subplot(1,5,4)

model_1303_cell06 = create_model("model_1303_cell06","2013_03_13_cell06_945_H42_05.ASC")
T,V = run_exp(model_1303_cell06,INJ,TSTOP)
plt.plot(T[0:int(BRIDGE_START/h.dt)+1],V[0:int(BRIDGE_START/h.dt)+1],linestyle='--',
												LineWidth=4,color=[185.0/255,82.0/159,159.0/255])
plt.plot(T[int(BRIDGE_END/h.dt)+1:v.size],V[int(BRIDGE_END/h.dt)+1:v.size],linestyle='--',
												LineWidth=4,color=[185.0/255,82.0/255,159.0/255],label = 'model')

plt.subplot(1,5,5)

model_1303_cell05 = create_model("model_1303_cell05","2013_03_13_cell05_675_H42_04.ASC")
T,V = run_exp(model_1303_cell05,INJ,TSTOP)
plt.plot(T[0:int(BRIDGE_START/h.dt)+1],V[0:int(BRIDGE_START/h.dt)+1],linestyle='--',
												LineWidth=4,color=[139.0/255,211.0/255,216.0/255])
plt.plot(T[int(BRIDGE_END/h.dt)+1:v.size],V[int(BRIDGE_END/h.dt)+1:v.size],linestyle='--',
												LineWidth=4,color=[139.0/255,211.0/255,216.0/255],label = 'model')


for ix in range(1,6):
	plt.subplot(1,5,ix)
	plt.axis([-20,100,E_PAS-0.5,E_PAS+2])
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	plt.xticks([])
	plt.yticks([])

plt.subplot(1,5,1)
plt.plot(np.array(range(50,91)),np.array(E_PAS+0.7*np.array([1 for i in range(41)])),'k',lw=2)
plt.text(58,E_PAS+0.6,'40 ms',fontsize=12)
    
plt.plot(np.array([50 for i in range(6)]),np.arange(E_PAS+0.7,E_PAS+1.3,0.1),'k',lw=2)
plt.text(40,E_PAS+1.05,'0.5 mV',rotation=90,fontsize=12)

fig = plt.figure(1)
fig.suptitle('Eyal et al. Fig. 1d',fontsize = 16)
plt.show()

