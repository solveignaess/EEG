########
#
# one demo case to shows the impact of cm on the chance to spike
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########

from neuron import h,gui
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
h.load_file("import3d.hoc")
h.load_file("nrngui.hoc")



NUMBER_OF_SYNAPSES =120
PLOT_SPIKES = 1


class params():
	pass

PARAMS = params()


#SIMULATION PARAMETERS
NUMBR_OF_RUNS = 1
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
WRITE_TO_FILE = 0
PRINT_SYNAPSES = 0

if len(sys.argv)>2:
	path1 = sys.argv[2]+"/syns"
else:
	path1 = "syns"
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
	cell.biophys()
	cell.active_biophys()

	return cell



cell = create_model("model_0603_cell08_cm045","2013_03_06_cell08_876_H41_05_Cell2.ASC")
soma = cell.soma[0]
PARAMS.SPINE_NECK_RA = soma.Ra


import Functions



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
			f.write("SEC_name\t")
	f.write("\n")
	return f





 # run the demo
def run(num_of_runs,num_of_syns,PARAMS,seed,f=None):
	random.seed(seed)
	for j in range(num_of_runs):
		sec_list,num_spikes,max_v= Functions.generateAndRunModel(num_of_syns,PARAMS,PLOT_SPIKES)
		if WRITE_TO_FILE:


			f.write(str(num_of_syns)+"\t"+str(num_spikes)+"\t"+str(max_v)+"\t")

			if PRINT_SYNAPSES:
				for s in range(num_of_syns):
					num_of_syns.write(sec_list[s].name()+"\t")
					
			
			f.write("\n")

		if (j%50 == 0):
			print "counter: ",j	
	

# for the demo we use constant seed
seed = 0

run(NUMBR_OF_RUNS,NUMBER_OF_SYNAPSES,PARAMS,seed)





