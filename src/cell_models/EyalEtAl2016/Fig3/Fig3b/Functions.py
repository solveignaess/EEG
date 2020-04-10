#!/usr/bin/env python

########
#
# Functions to use in random_syanpses. 
# These functions allow the creation od a model with X random synapses
# and then to run it and count the number of spikes.
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########

from neuron import h,gui
import numpy as np
import matplotlib.pyplot as plt
import random


# returns a random location on the tree (pointer to neuron section, and segment x)
# total_L: total dendritic length
# basal_L: total basal length 
def random_location(total_l,basal_l):
	rd=random.uniform(0,total_l)
	sum_l=0
	if rd<basal_l:
		for sec in h.cell.basal:
			
			if sum_l+sec.L>rd:
				x = (rd-sum_l)/float(sec.L)
				return_sec = sec
				return_x = x
				break
				
			sum_l+=sec.L
	else:
		sum_l = basal_l
		for sec in h.cell.apical:
			
			if sum_l+sec.L>rd:
				x = (rd-sum_l)/float(sec.L)
				return_sec = sec
				return_x = x
				break
			sum_l+=sec.L
	h.pop_section()
	return return_sec,return_x


# chosse # num_of_synaopses random synaptic locations on the tree
def fill_synapses_vectors(num_of_synapses):
	total_l = 0
	for sec in h.cell.basal:
		total_l+=sec.L
	basal_l = total_l
	for sec in h.cell.apical:
		total_l+=sec.L

	sec_list = []
	seg_list = []

	for i in range(num_of_synapses):
		temp_vec = random_location(total_l,basal_l)
		sec_list.append(temp_vec[0])
		seg_list.append(temp_vec[1])


	return sec_list,seg_list
	




	

# objref Synlist,Stim1,Conlist

# /* Generate the initial model for each run
# // distribute synapses and give them the values as decided in the config func
# // later version should include more suffisticated synaptic weight (mayve use rd to randomize from a vector of options) 
# */

def add_Spines(num_of_spines,sec_list,seg_list,PARAMS):
	hocL = h.List()
	for i in range(num_of_spines):
		sec_list[i].push()
		hocL.append(h.SectionRef())
		h.pop_section()
	hocVxs = h.Vector(seg_list) 
	h.cell.add_few_spines(hocL,hocVxs,PARAMS.SPINE_NECK_DIAM,PARAMS.SPINE_NECK_L,PARAMS.SPINE_HEAD_AREA,PARAMS.SPINE_NECK_RA)






def generateAndRunModel(num_of_synapses,PARAMS,plot_file = 0):

	
	h.cell.delete_spines()
	sec_list,seg_list = fill_synapses_vectors(num_of_synapses)
	if num_of_synapses>0:
		add_Spines(num_of_synapses,sec_list,seg_list,PARAMS)

	# Stimulus:
	Stim1 = h.NetStim()
	Stim1.interval=10000 
	Stim1.start=PARAMS.Spike_time
	Stim1.noise=0
	Stim1.number=1

	Synlist = []
	Conlist = []

	# distribute synapses on spine heads

	for i in range(num_of_synapses):
		
		Synlist.append(h.Exp2Syn(PARAMS.SPINE_HEAD_X,sec=sec_list[i]))
		Synlist.append(h.NMDA(PARAMS.SPINE_HEAD_X,sec=sec_list[i]))

					
		Synlist[2*i].e=PARAMS.E_SYN
		Synlist[2*i].tau1=PARAMS.TAU_1
		Synlist[2*i].tau2=PARAMS.TAU_2
		Synlist[2*i+1].e=PARAMS.E_SYN
		Synlist[2*i+1].tau_r_NMDA=PARAMS.TAU_1_NMDA
		Synlist[2*i+1].tau_d_NMDA=PARAMS.TAU_2_NMDA
		Synlist[2*i+1].n_NMDA = PARAMS.N_NMDA
		Synlist[2*i+1].gama_NMDA = PARAMS.GAMA_NMDA

			
		Conlist.append( h.NetCon(Stim1,Synlist[2*i])) 
		Conlist.append( h.NetCon(Stim1,Synlist[2*i+1]))
			
		Conlist[2*i].weight[0]=PARAMS.AMPA_W
		Conlist[2*i+1].weight[0]=PARAMS.NMDA_W

		Conlist[2*i].delay =0
		Conlist[2*i+1].delay =0
	
	V_model = h.Vector()
	tvec = h.Vector()
	apc = h.APCount(0.5,h.cell.soma[0])
	apc.thresh = 0               
	apc.time   = 10000000.      
	V_model.record(h.cell.soma[0](0.5)._ref_v)
	tvec.record(h._ref_t)
	
	h.v_init = PARAMS.V_INIT
	h.init(h.v_init)
	
	h.run()

	np_t = np.array(tvec)
	np_v = np.array(V_model)

	spike_x = np.where(np_t>0)[0][0] 

	if plot_file:
		plt.plot(np_t,np_v)
		plt.axis([0,h.tstop,-90,50])
		plt.show()

	return sec_list,apc.n,np_v[spike_x:].max()

	
	

