########
#
# this script read the results from num_syn_per_seg_NMDA_spike_time_threshold.txt for the six HL2/L3 cells
# and then creates a dict that map from secname and seg.x to its NMDA threshold
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########

import pandas as pnd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

path = "num_syn_per_seg_results/"
save_path = "../"
res_filename = "num_syn_per_seg_time_threshold_"
ouput_filename1 = "_seg_to_nmda_spike.pickle"


cells = ['cell0603_03','cell0603_08','cell0603_11',
			'cell1303_03','cell1303_05','cell1303_06']


THRESHOLD_MS = 20
for cell in cells:

	d = {}
	df = pnd.read_csv(path+res_filename+cell+".txt", header=None)
	table = np.array(df.iloc[:,2:])

	for ix,row in enumerate(table):
		sec_name = df.iloc[ix,0]
		dot_index = str.rfind(sec_name,".")
		sec_name = sec_name[dot_index+1:]
		seg = df.iloc[ix,1]
		seg = round(seg,3)
		above_threshold = np.where(row>THRESHOLD_MS)[0]
		if len(above_threshold) == 0:
			d[(sec_name,seg)] = None
		else:
			d[(sec_name,seg)] = above_threshold[0]+1

	with open(save_path+"dict_"+cell[4:]+ouput_filename1, 'wb') as handle:
	    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


