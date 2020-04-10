########
#
# This code generates figure 7B in Eyal et al 2017 that compares the IF curve of the six human L2/L3 models with the data
# IF curves were recorded in 25 human cells. 
# The models from Figure 7A were fitted also to result with a similar IF curve. 
# See details in the Manuscript
# The simulations may take about 10 minutes. 
# To a faster generation of the figures you can use the option of reading the model responses from txt files
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
import progressbar

# This function counts the number of spikes in trace V
def count_spikes(V,thresh,disp=0):
	apu = []
	apd = []
	firing = 0
	for j in range(len(V)):
		if firing and V[j] <= thresh:
			firing = 0
			apd.append(j)
		if not firing and V[j]>= thresh:
			firing = 1
			apu.append(j)

	if len(apu) != len (apd):
		print "Error: function ap is encountered an error"
	if disp:
		
		plt.plot(V)
		plt.scatter(apu,V[apu])
		plt.scatter(apd,V[apd])
		plt.show()

	return len(apu)



# Reads the data IF curves as recorded by Matthijs B. Verhoog and Guilherme Testa-Silva
# and plots them. 
# The IF curves were first linearly interpolated to increment of 1 pA
# and then normalized by the input resulted with 10 Hz firing rate
# In Eyal et al we show the normalize IF curves, 
# to see also the version which is not normalized, set show_not_normalized to True

def plot_data_IF(path = 'data/IF_data/',reference_F = 10, max_inp_for_mean = 4, show_not_normalized = False,
	thresh = 0,di_for_mean_i = 0.01):


	F_sum = [[] for i in range(int(max_inp_for_mean/di_for_mean_i)+1)]
	data_files = glob.glob(path+"*.mat")
	print "plots IF curve data"
	bar = progressbar.ProgressBar(max_value=len(data_files), widgets=[' [', progressbar.Timer(), '] ',
		progressbar.Bar(), ' (', progressbar.ETA(), ') ',    ])
	for jx,data_f in enumerate(data_files):
		data = sio.loadmat(data_f)
		first_pulse = data['abf']['first_pulse'][0][0][0][0]
		nrOfSweeps = data['abf']['nrOfSweeps'][0][0][0][0]
		pulse_delta = data['abf']['pulse_delta'][0][0][0][0]
		pulse_duration = data['abf']['pulse_duration'][0][0][0][0]
		I = first_pulse+np.array(range(nrOfSweeps))*pulse_delta
		traces = np.array(data['abf']['data'][0][0])
		F = []
		for i in range(traces.shape[1]):
			V = traces[:,i]
			n = count_spikes(V,thresh)
			F.append(n/(pulse_duration/1000.0))

		I_interp = np.array(range(I[0],I[-1]+1))
		F_interp = np.interp(I_interp,I,F)


		bar.update(jx)

		try:
			first_stim_above_reference_F = np.where(F_interp-reference_F>=0)[0][0]
			I_reference_F = I_interp[first_stim_above_reference_F]
			plt.figure(1)
			I_NORM = I/float(I_reference_F)
			plt.plot(I_NORM,F,c=[0.6,0.6,0.6])
		except:
			# This IF curve does not reach the reference spiking frequency, thus ignored.
			# for normalized firing rate of 10 Hz, happens with only 1 of the 25 cells.
			continue

		if show_not_normalized:
			plt.figure(2)
			plt.plot(I,F,c=[0.6,0.6,0.6])


		# another interpolation to ensure fair averaging between the different cells	
		I_norm_interp = np.arange(I_NORM[0],I_NORM[-1]+0.001,di_for_mean_i)
		F_interp = np.interp(I_norm_interp,I_NORM,F)

		# The indices of the first positive input and the maximal input that is smaller than 4 times the reference input
		# are found in order to calculate the mean IF curve for all the cells
		pos_inp = np.where(I_norm_interp>0)[0][0]
		if I_norm_interp[-1]>max_inp_for_mean:
			max_inp = np.where(I_norm_interp>max_inp_for_mean)[0][0]
		else:
			max_inp = len(I_norm_interp)-1
		for i,ix in enumerate(range(pos_inp,max_inp+1)):
			try:
				F_sum[i].append(F_interp[ix])
			except:
				import pdb
				pdb.set_trace()

	F_mean = np.array([np.mean(Fs) for Fs in F_sum])
	F_std = np.array([np.std(Fs) for Fs in F_sum])
	lower_bound = [max(0,m-s) for (m,s) in zip(F_mean,F_std)]
	I = np.arange(0,max_inp_for_mean+0.0001,di_for_mean_i)
	plt.figure(1)
	plt.plot(I,F_mean,'k')


# simulate the models and return their IF curves
# models were fitted using MOO (see Methods)
# and have active ion channels in their axon and soma
# This function generates the models and plot their responses to IF curve inputs. It may take few minutes to run
# if save_traces = True, the simulated tarces will be saved.


def simulate_models_IF(cells_list,models_dirs,models_path = "../ActiveModels/", morphs_path = "../Morphs/",
	save_traces = False,save_path = 'simulated_models_IF_Curve/',thresh = 0):
	h.load_file("import3d.hoc")
	h.steps_per_ms = 25
	h.dt = 1.0/h.steps_per_ms
	h.tstop = 1500
	h.celsius = 37
	h.v_init = -86


	for ix,cell in enumerate(cells_list):
		model = models_dirs[cell]['model']
		print "simulates model",model
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
		icl.amp = 0

		# Record the voltage at the soma
		Vsoma = h.Vector()
		Vsoma.record(HCell.soma[0](.5)._ref_v)
		tvec = h.Vector()
		tvec.record(h._ref_t)

		HCell.soma[0].push()
		MODEL_IF = []
		range_amps  = range(0,3000,100)
		bar = progressbar.ProgressBar(max_value=len(range_amps), widgets=[' [', progressbar.Timer(), '] ',
			progressbar.Bar(), ' (', progressbar.ETA(), ') ',    ])
		
		for jx,amp1000 in enumerate(range_amps):
			amp = amp1000/1000.0
			icl.amp = amp
			h.init(h.v_init)
			h.run()
			n = count_spikes(np.array(Vsoma),thresh)
			MODEL_IF.append((amp,n))

			bar.update(jx)


		MODEL_IF = np.array(MODEL_IF)

		models_dirs[cell]['I'] = MODEL_IF[:,0]
		models_dirs[cell]['F'] = MODEL_IF[:,1]
		
		if save_traces:
			if not os.path.exists(save_path):
				os.mkdir(save_path)
			np.savetxt(save_path+model+"IF.txt",np.array(MODEL_IF))


def read_models_IF(cells_list,models_dirs,saved_path = 'simulated_models_IF_Curve/'):
	for ix,cell in enumerate(cells_list):
		model = models_dirs[cell]['model']
		
		A = np.loadtxt(saved_path+model+"IF.txt")
		models_dirs[cell]['I'] = A[:,0]
		models_dirs[cell]['F'] = A[:,1] 



# plots the model responses to the same stimuli as in the data
# models can be simulated (default) or alternatively their voltage responses can be read from file. 
#To re-create the files - run 
def plot_model_IF_Curves(cells_list,models_dirs,reference_F = 10,read_from_file = False, show_not_normalized = False):


	for ix,cell in enumerate(cells_list):
		plt.figure(1)

		# normalize the IF curve of the model by the reference frequency
		I = models_dirs[cell]['I']
		F = models_dirs[cell]['F']
		interp_I = np.arange(int(I[0]),int(I[-1]),0.001)
		interp_F = np.interp(interp_I,I,F)

		F_ix_above_reference_F = np.where(interp_F-reference_F>=0)[0][0]
		NORM_I_reference = interp_I[F_ix_above_reference_F]
		NORM_I = interp_I/float(NORM_I_reference)
		plt.figure(1)
		plt.plot(NORM_I,interp_F,color = models_dirs[cell]['color'])

		if show_not_normalized:
			plt.figure(2)
			plt.plot(models_dirs[cell]['I']*1000,models_dirs[cell]['F'],color = models_dirs[cell]['color'])




def run_fig7B(cells_list,models_dirs,show_not_normalized=False, re_simulate_models= False, save_traces=False):
	plot_data_IF(show_not_normalized=show_not_normalized)
	if re_simulate_models:
		simulate_models_IF(cells_list,models_dirs,save_traces=save_traces)
	else:
		read_models_IF(cells_list,models_dirs)
	plot_model_IF_Curves(cells_list,models_dirs,show_not_normalized=show_not_normalized)








cells_list = ['cell0603_03','cell0603_08','cell0603_11','cell1303_03','cell1303_05','cell1303_06']
models_dirs = {}
models_dirs['cell0603_03'] = {'model':'cell0603_03_model_476','morph':'2013_03_06_cell03_789_H41_03.asc',
								'color':[0.0/255.0,0.0/255,255.0/255.0]}
models_dirs['cell0603_08'] = {'model':'cell0603_08_model_602','morph':'2013_03_06_cell08_876_H41_05_Cell2.asc',
								'color':[0.0/255.0,161.0/255,75.0/255.0]}

models_dirs['cell0603_11'] = {'model':'cell0603_11_model_937','morph':'2013_03_06_cell11_1125_H41_06.asc',
								'color':[237.0/255,31.0/255,36.0/255]}
models_dirs['cell1303_03'] = {'model':'cell1303_03_model_448','morph':'2013_03_13_cell03_1204_H42_02.asc',
								'color':[255.0/255,165.0/255.0,0.0/255.0]}
models_dirs['cell1303_05'] = {'model':'cell1303_05_model_643','morph':'2013_03_13_cell05_675_H42_04.asc',
								'color':[28.0/255,135.0/255,160.0/255]}
models_dirs['cell1303_06'] = {'model':'cell1303_06_model_263','morph':'2013_03_13_cell06_945_H42_05.asc',
								'color':[238.0/255,130.0/255,238.0/255]}

run_fig7B(cells_list,models_dirs)


