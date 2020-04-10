########
#
# This code generates figure 7A in Eyal et al 2017 that compares the spike response of the six human L2/L3 models with the data
# At least 10 repetitions of a spike trains of about 10 Hz were recorded in human L2/L3 neurons (see data)
# Based on this data and the 3d reconstructions of the morphologies of these cells
# We constructed active models for all the six neurons.
# The models were fitted using Multiple Objective Optimization
# See details in the Manuscript
# The simulations may take few minutes. 
# To a faster generation of the figures you can use the option of running the model responses from files
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########


import os

os.system('nrnivmodl ../ActiveMechanisms/')

import scipy.io as sio
import glob
import numpy as np

from neuron import h, gui
import matplotlib.pyplot as plt
import math
# import progressbar

# Read the data spikes as recorded by Matthijs B. Verhoog and plots them. 
# In each cell, at least 10 spikes were recorded. In Figure 7 we show only one example
# To plot all examples, call the function with plot_all_examples = True
def plot_data_spikes(cells_list,path = 'data/', plot_all_examples = False,JUNCTION_POTENTIAL = 16):


	IXs_of_traces_in_the_figures = [9,9,11,6,3,6]
	
	for ix,cell in enumerate(cells_list):
		traces = glob.glob(path+cell+"/*.txt")
		A = np.loadtxt(traces[IXs_of_traces_in_the_figures[ix]])
		plt.figure(ix)
		plt.plot(A[:,0],A[:,1]-JUNCTION_POTENTIAL,c='k')
		if plot_all_examples:
			print("plots data for", cell)
			# bar = progressbar.ProgressBar(max_value=len(traces), widgets=[' [', progressbar.Timer(), '] ',
			# 	progressbar.Bar(), ' (', progressbar.ETA(), ') ',    ])
			plt.figure(len(cells_list)+ix)
			
			for jx,trace_file in enumerate(traces):
				A = np.loadtxt(trace_file)
				plt.plot(A[:,0],A[:,1]-JUNCTION_POTENTIAL,c=[0.6,0.6,0.6])

				# bar.update(jx)


# simulates the models and return their voltage traces
# models were fitted using MOO (see Methods) and have active ion channels in their axon and soma
# This function generates the models and plot their response to the same stimulus as in the data
# if save_traces = True, the simulated traces will be saved.

def simulate_models(cells_list,models_dirs,models_path = "../ActiveModels/", morphs_path = "../Morphs/",
	save_traces = False,save_path = 'simulated_models_spike_trains/'):
	h.load_file("import3d.hoc")
	h.steps_per_ms = 100
	h.dt = 1.0/h.steps_per_ms
	h.tstop = 1500
	h.celsius = 37
	h.v_init = -86




	for ix,cell in enumerate(cells_list):
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


# Read model traces that were already simulated previously
def read_model_traces(cells_list,models_dirs,saved_path = 'simulated_models_spike_trains/'):
	for ix,cell in enumerate(cells_list):
		model = models_dirs[cell]['model']
		amp = models_dirs[cell]['stim_amp']
		A = np.loadtxt(saved_path+model+"_sim_"+str(int(amp*1000))+'pA.txt')
		models_dirs[cell]['t'] = A[:,0]
		models_dirs[cell]['v'] = A[:,1]

	

# plot the model responses to the same stimuli as in the data
# models can be simulated (default) or alternatively their voltage responses can be read from file. 
# To re-create the files - run simulate_models with save_traces = True

def plot_model_spikes(cells_list,models_dirs,read_from_file = False,plot_all_examples = False):


	for ix,cell in enumerate(cells_list):
		plt.figure(ix)
		plt.plot(models_dirs[cell]['t'],models_dirs[cell]['v'],color = models_dirs[cell]['color'])

		if plot_all_examples:
			plt.figure(len(cells_list)+ix)
			plt.plot(models_dirs[cell]['t'],models_dirs[cell]['v'],color = models_dirs[cell]['color'])



def run_fig7A(cells_list,models_dirs,plot_all_examples = False,re_simulate_models = False,save_traces= False):
	'''
	Runs and plots fig7A
	cells_list -- The cells to run and plot
	models_dirs -- a dictionary including the model parameters (template file, morphology,stimulus amplitude)
	plot_all_examples -- To cpmpare the model repsonse with all spike trains in the data
	re_simulate_models -- re-run the simulations with the models in cells_list
	save_traces -- save the simulations output to file
	'''
	plot_data_spikes(cells_list,plot_all_examples=plot_all_examples)
	if re_simulate_models:
		simulate_models(cells_list,models_dirs,save_traces=save_traces)
	else:
		read_model_traces(cells_list,models_dirs)
	plot_model_spikes(cells_list,models_dirs,plot_all_examples=plot_all_examples)







cells_list = ['cell0603_03','cell0603_08','cell0603_11','cell1303_03','cell1303_05','cell1303_06']
models_dirs = {}
models_dirs['cell0603_03'] = {'model':'cell0603_03_model_476','morph':'2013_03_06_cell03_789_H41_03.asc',
								'stim_amp':0.7,'color':[0.0/255.0,0.0/255,255.0/255.0]}
models_dirs['cell0603_08'] = {'model':'cell0603_08_model_602','morph':'2013_03_06_cell08_876_H41_05_Cell2.asc',
								'stim_amp':0.7,'color':[0.0/255.0,161.0/255,75.0/255.0]}

models_dirs['cell0603_11'] = {'model':'cell0603_11_model_937','morph':'2013_03_06_cell11_1125_H41_06.asc',
								'stim_amp':0.7,'color':[237.0/255,31.0/255,36.0/255]}
models_dirs['cell1303_03'] = {'model':'cell1303_03_model_448','morph':'2013_03_13_cell03_1204_H42_02.asc',
								'stim_amp':0.55,'color':[255.0/255,165.0/255.0,0.0/255.0]}
models_dirs['cell1303_05'] = {'model':'cell1303_05_model_643','morph':'2013_03_13_cell05_675_H42_04.asc',
								'stim_amp':0.55,'color':[28.0/255,135.0/255,160.0/255]}
models_dirs['cell1303_06'] = {'model':'cell1303_06_model_263','morph':'2013_03_13_cell06_945_H42_05.asc',
								'stim_amp':0.6,'color':[238.0/255,130.0/255,238.0/255]}

run_fig7A(cells_list,models_dirs)