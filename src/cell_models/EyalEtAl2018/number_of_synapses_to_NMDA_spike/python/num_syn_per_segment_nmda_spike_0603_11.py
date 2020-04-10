#!/usr/bin/env python

######
#
# This code goes over a model and searches how many synapses are required 
# in order to generate NMDA spike in each of the compartments of the model
# For each compartment, it activates between 1 and 50 synapses and measure how much time
# the local voltage was above -40 mV. 
# In Eyal et al, we defined NMDA spikes as events where the local voltage is above -40 for at least 20 ms
#
# As this script runs over all segments in the model and run 50 simulations in all of them
# it may take couple of hours to complete
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
######





import os

os.system('nrnivmodl ../../mechanisms/')

import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
from neuron import h,gui

h.load_file("nrngui.hoc")
h.load_file("import3d.hoc")


h("objref cell, tobj")
morph_file = "../../Morphs/2013_03_06_cell11_1125_H41_06.ASC"
model_file = "cell0603_11_model_cm_0_44"
model_path = "../../PassiveModels/"
h.load_file(model_path+model_file+".hoc")
h.execute("cell = new "+model_file+"()") 

nl = h.Import3d_Neurolucida3()
nl.quiet = 1
nl.input(morph_file)
imprt = h.Import3d_GUI(nl,0)
imprt.instantiate(h.cell)	
HCell = h.cell
HCell.create_model()
HCell.geom_nseg()
HCell.biophys()



TAU_1_AMPA = 0.3
TAU_2_AMPA = 1.8
# properties of the model from figure 4:
TAU_1_NMDA = 8.019 
TAU_2_NMDA = 34.9884
N_NMDA = 0.28011
GAMMA_NMDA = 0.0765685
AMPA_W = 0.00073027
NMDA_W = 0.00131038
NMDA_W_BLOCKED = 0


E_SYN = 0
E_PAS = -86
Spike_time = 10
DELAY = 0
NUM_OF_SYNAPSES = 1
SPINE_HEAD_X = 1
SECTION_TERMINAL = 1

h.tstop = 100
V_INIT = E_PAS
CLUSTER_LENGTH = 20
VOLTAGE_THRESHOLD_FOR_TIME = -40


Stim1 = h.NetStim()
Stim1.interval=10000 
Stim1.start=Spike_time
Stim1.noise=0
Stim1.number=1


# Add num_of_synapses on section sec at segment x
# The spines are distributed in CLUSTER_LENGTH
def add_spines_on_seg(num_of_synapses,sec,x):
	Lsec = sec.L
	sref = h.SectionRef(sec=sec)
	xL = float(x)*Lsec
	maxX = xL+CLUSTER_LENGTH/2.0
	if xL+10>Lsec:
		maxX = Lsec
	minX = xL-10
	if xL-10<0:
		minX = 0

	local_vec_x = h.Vector()
	list_sref = h.List()
	for ix in range(num_of_synapses):
		list_sref.append(sref)
		local_vec_x.append(np.random.uniform(minX/Lsec,maxX/Lsec))

	HCell.add_few_spines(list_sref,local_vec_x,0.25,1.35,2.8,HCell.soma[0].Ra)

SynList = []
ConList = []

def set_AMPA_NMDA_synapses(num_of_synapses):
	
	del SynList[:]
	del ConList[:]
	
	for j in range(num_of_synapses):

			SynList.append(h.Exp2Syn(SPINE_HEAD_X,sec=HCell.spine[(j*2)+1]))
			ConList.append(h.NetCon(Stim1,SynList[-1]))
			SynList[-1].e=E_SYN
			SynList[-1].tau1=TAU_1_AMPA
			SynList[-1].tau2=TAU_2_AMPA
			ConList[-1].weight[0]= AMPA_W
			ConList[-1].delay = DELAY


	for j in range(num_of_synapses):
		
		SynList.append(h.NMDA(SPINE_HEAD_X,sec=HCell.spine[(j*2)+1]))
		ConList.append(h.NetCon(Stim1,SynList[-1]))
		SynList[-1].e=E_SYN
		SynList[-1].tau_r_NMDA=TAU_1_NMDA
		SynList[-1].tau_d_NMDA=TAU_2_NMDA
		ConList[-1].weight[0] = NMDA_W
		ConList[-1].delay = DELAY
		SynList[-1].n_NMDA = N_NMDA
		SynList[-1].gama_NMDA = GAMMA_NMDA


output_file_t_thresh = "num_syn_per_seg_results/num_syn_per_seg_NMDA_spike_time_threshold" + model_file[:11] + ".txt"	

s_v_thresh = ""
s_time_thresh = ""
k = 0
for sec in list(HCell.basal)+list(HCell.apical):
	k+=1
	sref = h.SectionRef(sec=sec)

	for seg in list(sec)+[sec(1)]:
		if seg.x == 0:
			continue

	
		maxV = []
		time_above_threshold = []
		for S in range(1,51):
			HCell.delete_spine()
			dendV = h.Vector()
			tV = h.Vector()
			add_spines_on_seg(S,sec,seg.x)
			set_AMPA_NMDA_synapses(S)	
			dendV.record(sec(seg.x)._ref_v)
			tV.record(h._ref_t)

			h.v_init = V_INIT
			h.init(h.v_init)
			h.run()
			MAXV = dendV.max()
			maxV.append(MAXV)
			dendV = np.array(dendV)
			above_threshold = np.where(dendV>VOLTAGE_THRESHOLD_FOR_TIME)
			above_threshold = above_threshold[0]
			

			if len(above_threshold) ==0:
				time_above_threshold.append(0)
			else:
				tV = np.array(tV)
				st = tV[above_threshold[0]]
				en = start = tV[above_threshold[-1]]
				time_above_threshold.append(en-st)
			
			if S ==50:
				print seg.x,len(above_threshold),MAXV


		s_time_thresh += sec.hname()+","+str(seg.x)+","+",".join([str(t) for t in time_above_threshold])+"\n"

with open(output_file_t_thresh,'w') as f:
	f.write(s_time_thresh)



