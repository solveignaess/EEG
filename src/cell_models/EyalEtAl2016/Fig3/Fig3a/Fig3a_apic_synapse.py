########
#
# This code generate Fig 3A for the apical synapse in Eyal ey al. 2016
# you can test other cases by changing dendritic locations in the parameters 
# 
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########



from neuron import h,gui
import numpy as np
import matplotlib.pyplot as plt
import os
h.load_file("import3d.hoc")
h.load_file("nrngui.hoc")

CM045_COLOR  = 'r'
CM09_COLOR = 'b'


# A function that generate neuron model based on template model_name and morph_file 

def create_model(model_name,morph_file):
	h("objref cell, tobj")
	morph_file = "../../morphs/"+morph_file
	model_path = "../../ActiveModels/"
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

	return cell

# Model parameters
StepDist = 60
F = 1.9
SPINE_NECK_DIAM =0.25 
SPINE_NECK_L = 1.35
SPINE_HEAD_AREA = 2.8
cell = create_model("model_0603_cell08_cm045","2013_03_06_cell08_876_H41_05_Cell2.ASC")
soma = cell.soma[0]
SPINE_NECK_RA = soma.Ra

E_SYN = 0
TAU_1 = 0.3
TAU_2 = 1.8
TAU_1_NMDA = 3
TAU_2_NMDA = 75
N_NMDA = 0.280112
GAMA_NMDA = 0.08
NMDA_W = 0.0014
AMPA_W = 0.0007

# constants:
APIC = "APIC"
BASAL = "BASAL"

# running constants:
DEND_TREE = APIC
DEND_SEC = 40
Spike_time =200
TSTOP = 400
V_INIT =  -86
SPINE_HEAD_X =1
DEND_X = 1
DT = 1.0/100



Stim1 = h.NetStim(0.5,sec=soma)
Stim1.interval=10000 #ms
Stim1.start=Spike_time
Stim1.noise=0
Stim1.number=1

SynList = []
ConList = []

# Add human spines to the human model in the locations of random synapses
def add_Spines(num_of_spines,syn_trees,syn_secs,syn_segs):
	hocL = h.List()
	for i in range(num_of_spines):
		if syn_trees[i] == APIC:
			cell.apic[syn_secs[i]].push()
		else:
			cell.dend[syn_secs[i]].push()
		hocL.append(h.SectionRef())
		h.pop_section()
	hocVxs = h.Vector(syn_segs) 
	cell.add_few_spines(hocL,hocVxs,SPINE_NECK_DIAM,SPINE_NECK_L,SPINE_HEAD_AREA,SPINE_NECK_RA)

# Add AMPA and NMDAsynapse in the head of the spine
def add_AMPA_NMDA_synapse(num_of_synapses):
	del SynList[:]
	del ConList[:]
	for j in range(num_of_synapses):
		SynList.append(h.Exp2Syn(SPINE_HEAD_X,sec=cell.spine[(j*2)+1]))
		ConList.append(h.NetCon(Stim1,SynList[-1]))
		SynList[-1].e = E_SYN
		SynList[-1].tau1 = TAU_1
		SynList[-1].tau2 = TAU_2
		ConList[-1].weight[0] = AMPA_W
		ConList[-1].delay = 0
		SynList.append(h.NMDA(SPINE_HEAD_X,sec=cell.spine[(j*2)+1]))
		ConList.append(h.NetCon(Stim1,SynList[-1]))
		SynList[-1].e = E_SYN
		SynList[-1].tau_r_NMDA=TAU_1_NMDA
		SynList[-1].tau_d_NMDA=TAU_2_NMDA
		SynList[-1].n_NMDA = N_NMDA
		SynList[-1].gama_NMDA = GAMA_NMDA
		ConList[-1].weight[0] = NMDA_W
		ConList[-1].delay = 0

# record the voltage in the model's spine, dendrite and soma
def recording_vecs():
	Tvec = h.Vector()
	Tvec.record(h._ref_t)
	Vsoma = h.Vector()
	Vsoma.record(soma(0.5)._ref_v)
	Vdend = h.Vector()
	if DEND_TREE == APIC:
		Vdend.record(cell.apic[DEND_SEC](DEND_X)._ref_v)
	else:
		Vdend.record(cell.dend[DEND_SEC](DEND_X)._ref_v)
	Vspine = h.Vector()
	Vspine.record(cell.spine[1](SPINE_HEAD_X)._ref_v)

	return Tvec,Vsoma,Vdend,Vspine

def Hoc_vecs_to_np(*args):
	l = []
	for vec in args:
		l.append(np.array(vec))
	return l


# Run the experminet. 
# First measure the voltage for the human model (Cm = 0.45)
# Then change the Cm by a factor of 2 to present the impact of the specific capacitance on synaptic charge transfer 
def run_exp():
	synapses = 1
	cell.delete_spines()
	syn_trees = []
	syn_sections = []
	syn_segments = []


	syn_trees.append(DEND_TREE)
	syn_sections.append(DEND_SEC)
	syn_segments.append(DEND_X)

	add_Spines(synapses,syn_trees,syn_sections,syn_segments)
	add_AMPA_NMDA_synapse(synapses)

	h.tstop = TSTOP
	h.steps_per_ms = 1/DT
	h.dt = DT
	
	Tvec,Vsoma,Vdend,Vspine = recording_vecs()
	h.v_init = V_INIT
	h.init(V_INIT)
	h.run()

	# Add record
	np_t,np_soma,np_dend,np_spine = Hoc_vecs_to_np(Tvec,Vsoma,Vdend,Vspine)

	np_t = np_t - Spike_time
	plt.figure(1)
	plt.subplot(121)
	plt.plot(np_t,np_soma,color=CM045_COLOR,label='Cm=0.45')
	plt.subplot(122)
	plt.plot(np_t,np_spine,color=CM045_COLOR,label='Cm=0.45')

	cell.change_cm(2)

	Tvec,Vsoma,Vdend,Vspine = recording_vecs()
	h.v_init = V_INIT
	h.init(V_INIT)
	h.run()

	# Add record
	np_t,np_soma,np_dend,np_spine = Hoc_vecs_to_np(Tvec,Vsoma,Vdend,Vspine)
	np_t = np_t - Spike_time

	plt.figure(1)
	plt.subplot(121)
	plt.plot(np_t,np_soma,color=CM09_COLOR,label='Cm=0.9')
	plt.axis([-10,190,-86.11,-86.06])
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	plt.xlabel('Time (ms)',fontsize = 14)
	plt.ylabel('Voltage (mV)',fontsize = 14)
	plt.title('EPSP in the soma',fontsize = 16)
	plt.ticklabel_format(useOffset=False, style='plain')
	plt.subplot(122)

	plt.plot(np_t,np_spine,color=CM09_COLOR,label='Cm=0.9')
	plt.axis([-10,190,-87,-67])
	plt.legend(loc='best', fancybox=True, framealpha=0.5)
	plt.xlabel('Time (ms)',fontsize = 14)
	plt.ylabel('Voltage (mV)',fontsize = 14)
	plt.title('EPSP in the spine head',fontsize = 16)
	plt.ticklabel_format(useOffset=False, style='plain')

	plt.tight_layout()
	




run_exp()
plt.show()









