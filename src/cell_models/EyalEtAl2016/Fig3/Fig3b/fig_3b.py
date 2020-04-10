########
#
# This code generate Fig 3b in Eyal ey al. 2016
# Here we quantify how many excitatory synapses should be simultaneously activated 
# for initiating a somatic Na+ spike in our model for HL2/3 PCs 
#
# This script use the results of run_cm045 and run_cm09
# 
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########


import os
import numpy as np
import glob
import matplotlib.pyplot as plt

# results of run_cm045 and run_cm09
RES_CM_045 = "Mon_Apr_11_11_35_14_2016"
RES_CM_09 = "Mon_Apr_11_11_35_26_2016"
DIR_CM_09 = 'DIST_09_SYNC'
DIR_CM_045 = 'DIST_045_SYNC'


NUM_SYNS = 301

mean_voltage_cm_045 = []
mean_spike_prob_cm_045 = []
std_voltage_cm_045 = []
std_spike_prob_cm_045 = []

# read the results for the cm=0.45 model
os.chdir(DIR_CM_045+'/'+RES_CM_045)
for s in range(NUM_SYNS):
	os.chdir("syns_"+(str(s)))
	filenames = glob.glob('sync_*')
	if len(filenames)>1:
		print "TOO MUICH FILES IN DIR", os.getcwd()
	try:
		RES = np.loadtxt(filenames[0],skiprows=1)
	except: 
		print "error in ",os.getcwd()
		os.chdir('../')
		continue
	spike_prob = RES[:,1]>0
	spike_prob=spike_prob.astype(int)
	mean_voltage_cm_045.append(RES[:,2].mean())
	std_voltage_cm_045.append(RES[:,2].std())
	mean_spike_prob_cm_045.append(spike_prob.mean())
	std_spike_prob_cm_045.append(spike_prob.std())
	os.chdir("../")

os.chdir("../../")


np_mean_v_cm_045 = np.array(mean_voltage_cm_045)
np_std_v_cm_045 = np.array(std_voltage_cm_045)
np_mean_sp_prob_cm_045 = np.array(mean_spike_prob_cm_045)
np_std_sp_prob_cm_045 = np.array(std_spike_prob_cm_045)


mean_voltage_cm_09 = []
mean_spike_prob_cm_09 = []
std_voltage_cm_09 = []
std_spike_prob_cm_09 = []

# read the results for the cm=0.9 model
os.chdir(DIR_CM_09+'/'+RES_CM_09)
for s in range(NUM_SYNS):
	os.chdir("syns_"+(str(s)))
	filenames = glob.glob('sync_*')
	if len(filenames)>1:
		print "TOO MUICH FILES IN DIR", os.getcwd()
	try:
		RES = np.loadtxt(filenames[0],skiprows=1)
	except: 
		print "error in ",os.getcwd()
		os.chdir('../')
		continue
	spike_prob = RES[:,1]>0

	spike_prob=spike_prob.astype(int)
	mean_voltage_cm_09.append(RES[:,2].mean())
	std_voltage_cm_09.append(RES[:,2].std())
	mean_spike_prob_cm_09.append(spike_prob.mean())
	std_spike_prob_cm_09.append(spike_prob.std())
	os.chdir("../")

os.chdir("../../")


np_mean_v_cm_09 = np.array(mean_voltage_cm_09)
np_std_v_cm_09 = np.array(std_voltage_cm_09)
np_mean_sp_prob_cm_09 = np.array(mean_spike_prob_cm_09)
np_std_sp_prob_cm_09 = np.array(std_spike_prob_cm_09)

np_syns = np.array(range(NUM_SYNS))
# just for plotting - as spike prob can't be really below 0 or above 1

max_std_09 = np.array([min(np_mean_sp_prob_cm_09[i]+np_std_sp_prob_cm_09[i],1) for i in range(np_mean_sp_prob_cm_09.size)] )
min_std_09 = np.array([max(np_mean_sp_prob_cm_09[i]-np_std_sp_prob_cm_09[i],0) for i in range(np_mean_sp_prob_cm_09.size)] )

max_std_045 = np.array([min(np_mean_sp_prob_cm_045[i]+np_std_sp_prob_cm_045[i],1) for i in range(np_mean_sp_prob_cm_045.size)] )
min_std_045 = np.array([max(np_mean_sp_prob_cm_045[i]-np_std_sp_prob_cm_045[i],0) for i in range(np_mean_sp_prob_cm_045.size)] )


fig, ax = plt.subplots(1)
ax.fill_between(np_syns, min_std_09, max_std_09, facecolor='blue', alpha=0.5)
ax.fill_between(np_syns, max_std_045, min_std_045, facecolor='red', alpha=0.5)
ax.plot(np_syns, np_mean_sp_prob_cm_09, lw=2, color='blue',label ='Cm = 0.9')
ax.plot(np_syns, np_mean_sp_prob_cm_045, lw=2, color='red',label ='Cm = 0.45')

ax.set_xlabel('Spike probability',fontsize=14)
ax.set_ylabel('Number of synapses',fontsize=14)
ax.legend(loc='best', fancybox=True, framealpha=0.5)
plt.title('Eyal et al. Fig. 3c',fontsize=16)
ax.axis([0,220,0,1])

plt.show()




