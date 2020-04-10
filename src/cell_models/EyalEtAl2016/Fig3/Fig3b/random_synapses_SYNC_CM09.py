#!/usr/bin/env python

########
#
# This code generate a model with cm = 0.9
# then for 1000 trials it distributes X (argv[1]) ranodm synapses on the tree
# and print to file the peak voltage and number of spikes in each trial
# 
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########


from neuron import h,gui
import numpy as np
import sys
import random
h.load_file("import3d.hoc")
h.load_file("nrngui.hoc")

FUNCTIONS_LOCATION = os.getcwd()
sys.path.insert(0, FUNCTIONS_LOCATION)
import Functions

# Number of synapses should be suplllied in sys.argv[1]
NUMBER_OF_SYNAPSES =110
if len(sys.argv)>1:
	NUMBER_OF_SYNAPSES = int(sys.argv[1])



class params():
	pass
PARAMS = params()


#SIMULATION PARAMETERS
NUMBR_OF_RUNS = 1000
PARAMS.Spike_time =131.4 
h.cvode_active(0)
h.steps_per_ms = 100
h.dt = 1/h.steps_per_ms
h.tstop = PARAMS.Spike_time+150
#MODEL PARAMETERS:

PARAMS.SPINE_NECK_DIAM =0.25 
PARAMS.SPINE_NECK_L = 1.35
PARAMS.SPINE_HEAD_AREA = 2.8


PARAMS.E_SYN = 0
PARAMS.TAU_1 = 0.3
PARAMS.TAU_2 = 1.8
PARAMS.TAU_1_NMDA = 3
PARAMS.TAU_2_NMDA = 75
PARAMS.N_NMDA = 0.280112
PARAMS.GAMA_NMDA = 0.08
PARAMS.NMDA_W = 0.0014
PARAMS.AMPA_W = 0.0007
PARAMS.V_INIT = -86
PARAMS.SPINE_HEAD_X=1

# Saving files parameters:
WRITE_TO_FILE = 1
PRINT_SYNAPSES = 0

# The path to the directory of this experiment
if len(sys.argv)>2:
	path1 = sys.argv[2]+"/syns"
else:
	path1 = "temp_syns/syns"
path2 = "sync_stim"

seed_filename = "seed/seed.txt"

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



cell = create_model("model_0603_cell08_cm09","2013_03_06_cell08_876_H41_05_Cell2.ASC")
soma = cell.soma[0]
PARAMS.SPINE_NECK_RA = soma.Ra


# get new seed from the seed file
def get_seed(seed_filename):
	with open(seed_filename) as f:
		line = f.readline()
		while line:
			seed = int(line.strip())
			line = f.readline()
	with open(seed_filename,'a') as f:
		f.write('\n'+str(seed+1))
	return seed



def open_writing_file(seed,num_of_syns):
	filename = path1+"_"+str(num_of_syns)+"/"+path2+'_seed_'+str(seed)+".txt"
	f = open(filename,'w')
	f.write("NUMBER_OF_SYNAPSES\tAPC_N\tMAX_VOLTAGE\t")
	if (PRINT_SYNAPSES):
		for i in range(num_of_syns):
			f.write("syn_loc_"+str(i)+"\t")
	f.write("\n")
	return f





 # run the simulation for num_of_runs with num_of_synapses 
 # and print the result to f 
def run_runs(num_of_runs,num_of_syns,PARAMS,seed,f):
	random.seed(seed)
	for j in range(num_of_runs):
		sec_list,num_spikes,max_v= Functions.generateAndRunModel(num_of_syns,PARAMS)
		if WRITE_TO_FILE:


			f.write(str(num_of_syns)+"\t"+str(num_spikes)+"\t"+str(max_v)+"\t")

			if PRINT_SYNAPSES:
				for s in range(num_of_syns):
					f.write(sec_list[s].name()+"\t")
					
			
			f.write("\n")
		if (j%50 == 0):
			print "counter: ",j	
	

seed = get_seed(seed_filename)
print "Seed: ",seed
if WRITE_TO_FILE:
	f = open_writing_file(seed,NUMBER_OF_SYNAPSES)
else:
	f = None
run_runs(NUMBR_OF_RUNS,NUMBER_OF_SYNAPSES,PARAMS,seed,f)

f.close()




