########
#
# Reads the txtfiles from num_syn_per_seg_results
# and analyze them: How many synapses are required to generate NMDA spike in average.
# write the results:
# to nmda_spike_time_threshold.txt (for any dednritic branch)
# and nmda_spike_time_threshold_terminals.txt (for the analysis of dednritic terminals)
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########


import pandas as pnd
import os
import numpy as np
import matplotlib.pyplot as plt




cells = ['cell0603_03','cell0603_08','cell0603_11',
			'cell1303_03','cell1303_05','cell1303_06']

path = 'num_syn_per_seg_results/'
all_segments_time_threshold_file = 'num_syn_per_seg_time_threshold_'
terminals_time_threshold_file = 'num_syn_per_seg_time_threshold_terminals_'


# go over the above files and see for each segment and for each terminal find its synaptic threshold
def nmda_spike_conditions_defined_by_time_threshold(save_to_file=1,all_segs = True,terminals = True,
		filename_all_segs = 'nmda_spike_time_threshold.txt',
		filename_terminals = 'nmda_spike_time_threshold_terminals.txt',threshold_ms = 20):
	
	SEC_NAME = 0
	SEG_NAME = 1
	NUM_SYN_SHIFT = 2

	# running over the files of all segments
	if all_segs: 
		counted_segments_reached_the_threshold_list = []
		counted_total_segments_list = []
		synapses_per_spike_list = []

		for cell in cells:
			synapses_per_spike_cell = []
			
			df = pnd.read_csv(path+all_segments_time_threshold_file+cell+".txt", header=None)
			
			table = np.array(df.iloc[:,2:])
			counted_total_segments=np.shape(table)[0]
			counted_segments_reached_the_threshold = 0
			for ix,row in enumerate(table):
				above_threshold = np.where(row>threshold_ms)[0]
				
				if len(above_threshold) == 0:
					continue
				counted_segments_reached_the_threshold+=1
				synapses_per_spike_cell.append(above_threshold[0]+1)
			synapses_per_spike_list.append(synapses_per_spike_cell)
			counted_total_segments_list.append(counted_total_segments)
			counted_segments_reached_the_threshold_list.append(counted_segments_reached_the_threshold)

		synapses_per_spike = [s for sublist in synapses_per_spike_list for s in sublist]

		precentage_of_seg_reached_threshold = np.sum(counted_segments_reached_the_threshold_list)/float(np.sum(counted_total_segments_list))
		if save_to_file:
			with open(filename_all_segs,'w') as f:
				f.write("when nmda spike is defined as being above -40 mV for "+str(threshold_ms) + " ms\n")
				f.write("nmda spike was created in "+"%.3f"%precentage_of_seg_reached_threshold+ " of the segments\n")
				f.write("nmda spike was created with "+"%.3f"%np.mean(synapses_per_spike)+ " +- "+"%.3f"%np.std(synapses_per_spike)+" synapses\n")

				f.write("\n")
				# values for each cells
				for ix,cell in enumerate(cells):

					
					perc = counted_segments_reached_the_threshold_list[ix]/float(counted_total_segments_list[ix])
					m = np.mean(synapses_per_spike_list[ix])
					s = np.std(synapses_per_spike_list[ix])
					f.write(cell+": "+"%.3f"%(perc*100)+"%, "+"%.3f"%m+" +- "+"%.3f"%s+"\n")


	#files that have the data only from terminals
	if terminals: 

		total_basal_segments_list = []
		basal_segments_reached_the_threshold_list = []
		synapses_per_spike_basal_list = []
		total_apical_segments_list = []
		apical_segments_reached_the_threshold_list = []
		synapses_per_spike_apical_list = []



		for cell in cells:
			total_basal_segments = 0 
			basal_segments_reached_the_threshold = 0 
			synapses_per_spike_basal = []
			total_apical_segments = 0
			apical_segments_reached_the_threshold = 0
			synapses_per_spike_apical = []

			df = pnd.read_csv(path+terminals_time_threshold_file+cell+".txt", header=None)			
			table = np.array(df.iloc[:,2:])
			for ix,row in enumerate(table):
				if str.find(df.iloc[:,0][ix],'dend')>=0:
					total_basal_segments += 1
					above_threshold = np.where(row>threshold_ms)[0]
					if len(above_threshold) == 0:
						continue
					basal_segments_reached_the_threshold+=1
					synapses_per_spike_basal.append(above_threshold[0]+1)
				elif str.find(df.iloc[:,0][ix],'apic')>=0:
					total_apical_segments += 1
					above_threshold = np.where(row>threshold_ms)[0]
					if len(above_threshold) == 0:
						continue
					apical_segments_reached_the_threshold+=1
					synapses_per_spike_apical.append(above_threshold[0]+1)

			total_basal_segments_list.append(total_basal_segments)
			basal_segments_reached_the_threshold_list.append(basal_segments_reached_the_threshold)
			synapses_per_spike_basal_list.append(synapses_per_spike_basal)
			total_apical_segments_list.append(total_apical_segments)
			apical_segments_reached_the_threshold_list.append(apical_segments_reached_the_threshold)
			synapses_per_spike_apical_list.append(synapses_per_spike_apical)


		synapses_per_spike_basal = [s for sublist in synapses_per_spike_basal_list for s in sublist]
		synapses_per_spike_apical = [s for sublist in synapses_per_spike_apical_list for s in sublist]

		precentage_of_seg_reached_threshold_basal = np.sum(basal_segments_reached_the_threshold_list)/float(np.sum(total_basal_segments_list))
		precentage_of_seg_reached_threshold_apical = np.sum(apical_segments_reached_the_threshold_list)/float(np.sum(total_apical_segments_list))
		precentage_of_seg_reached_threshold = (np.sum(basal_segments_reached_the_threshold_list)+np.sum(apical_segments_reached_the_threshold_list))/ \
							float(np.sum(total_basal_segments_list)+np.sum(total_apical_segments_list))
		
		if save_to_file:
			with open(filename_terminals,'w') as f:
				f.write("when nmda spike is defined as being above -40 mV for "+str(threshold_ms) + " ms\n")
				
				f.write("nmda spike was created in "+str(precentage_of_seg_reached_threshold)+ " of the terminals\n")
				f.write("nmda spike was created with "+str(np.mean(np.array(synapses_per_spike_basal+synapses_per_spike_apical)))+ " +- "+
					str(np.std(np.array(synapses_per_spike_basal+synapses_per_spike_apical)))+" synapses\n")

				f.write("\nbasal:\n")
				f.write("nmda spike was created in "+str(precentage_of_seg_reached_threshold_basal)+ " of the basal terminals\n")
				f.write("nmda spike was created with "+str(np.mean(synapses_per_spike_basal))+ " +- "+
					str(np.std(synapses_per_spike_basal))+" synapses\n")

				f.write("\napical:\n")
				f.write("nmda spike was created in "+str(precentage_of_seg_reached_threshold_apical)+ " of the apical terminals\n")
				f.write("nmda spike was created with "+str(np.mean(synapses_per_spike_apical))+ " +- "+
					str(np.std(synapses_per_spike_apical))+" synapses\n")

				# values for each cells
				f.write("\nall:\n")
				for ix,cell in enumerate(cells):
					
					perc = (basal_segments_reached_the_threshold_list[ix]+apical_segments_reached_the_threshold_list[ix])/ \
							float(total_basal_segments_list[ix]+total_apical_segments_list[ix])
					m = np.mean(synapses_per_spike_apical_list[ix]+synapses_per_spike_basal_list[ix])
					s = np.std(synapses_per_spike_apical_list[ix]+synapses_per_spike_basal_list[ix])
					f.write(cell+": "+"%.3f"%(perc*100)+"%, "+"%.3f"%m+" +- "+"%.3f"%s+"\n")

				f.write("\nbasal:\n")
				for ix,cell in enumerate(cells):
					
					perc = basal_segments_reached_the_threshold_list[ix]/ float(total_basal_segments_list[ix])
					m = np.mean(synapses_per_spike_basal_list[ix])
					s = np.std(synapses_per_spike_basal_list[ix])
					f.write(cell+": "+"%.3f"%(perc*100)+"%, "+"%.3f"%m+" +- "+"%.3f"%s+"\n")

				f.write("\napcial:\n")
				for ix,cell in enumerate(cells):
					
					perc = apical_segments_reached_the_threshold_list[ix]/ float(total_apical_segments_list[ix])
					m = np.mean(synapses_per_spike_apical_list[ix])
					s = np.std(synapses_per_spike_apical_list[ix])
					f.write(cell+": "+"%.3f"%(perc*100)+"%, "+"%.3f"%m+" +- "+"%.3f"%s+"\n")




nmda_spike_conditions_defined_by_time_threshold(save_to_file=1)


	
