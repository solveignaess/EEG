########
#
# This code generates a shorter version of figure 8A in Eyal et al 2017
# It simulates synchronous activation of synapses and the probability to spike as function of the number of activated synapses.
# Two cases are shown, the case of distributed activation and the case of clustered activation
# This is a short version, because in the paper much more cases and seeds per case where tested
#  
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########
import os

# os.system('nrnivmodl ../ActiveMechanisms/')

import numpy as np
from neuron import h, gui
import neuron
import matplotlib.pyplot as plt
import math
# import progressbar

seed = 100 
MAX_JITTER = 0 #synchronous activation
cluster_type = "sync_clusters"

NUMBER_OF_RUNS_DIST = 9
DISTRIB_SYNS_RANGE = [120]
NUMBER_OF_RUNS_CLUSTER = 10
CLUSTERS_RANGE = [0,2,4,5,6,7,8,10,12]

COLOR_CLUSTER = np.array([239,67,182])/256.0
COLOR_DISTRIBUTED = np.array([10,153,20])/256.0

model = "cell0603_11_model_937"
h.load_file("import3d.hoc")
neuron.load_mechanisms("../ActiveMechanisms/")
h.load_file("../ActiveModels/"+model+".hoc")
h("objref HCell")
h("HCell = new "+model+"()")
HCell = h.HCell
nl = h.Import3d_Neurolucida3()

# Creating the model
nl.quiet = 1
nl.input("../Morphs/2013_03_06_cell11_1125_H41_06.ASC")
imprt = h.Import3d_GUI(nl, 0)   
imprt.instantiate(HCell)    
HCell.indexSections(imprt)
HCell.geom_nsec()   
HCell.geom_nseg()
HCell.delete_axon()
HCell.insertChannel()
HCell.init_biophys()
HCell.biophys()

class config_params():
    pass

config = config_params()

config.CLUSTER_TYPE = None
config.TAU_1_AMPA = 0.3
config.TAU_2_AMPA = 1.8
# properties of the model from figure 4:
config.TAU_1_NMDA = 8.019 
config.TAU_2_NMDA = 34.9884
config.N_NMDA = 0.28011
config.GAMMA_NMDA = 0.0765685
config.AMPA_W = 0.00073027
config.NMDA_W = 0.00131038
config.NMDA_W_BLOCKED = 0
config.E_SYN = 0

config.Spike_time =131.4
config.SPINE_HEAD_X = 1
config.CLUSTER_L = 20
config.CLUSTER_SIZE = 20




h.steps_per_ms = 25
h.dt = 1.0/h.steps_per_ms
h.celsius = 37
h.v_init = -86

Stim1 = h.NetStim()
Stim1.interval=10000 
Stim1.start= config.Spike_time
Stim1.noise=0
Stim1.number=1

config.stim = Stim1

h.tstop = config.Spike_time+150


from Figure8Functions import *

rd = h.Random(seed)

spike_counts_distributed = {}
max_v_distributed = {}
print("simulating the distributed case:")
# bar = progressbar.ProgressBar(max_value=len(DISTRIB_SYNS_RANGE*NUMBER_OF_RUNS_DIST), widgets=[' [', progressbar.Timer(), '] ',
#         progressbar.Bar(), ' (', progressbar.ETA(), ') ',    ])

for jx,number_of_synapses in enumerate(DISTRIB_SYNS_RANGE):
    spike_prob_list = []
    spike_counts = []
    max_v_list = []
    for rep in range(NUMBER_OF_RUNS_DIST):
        # randomly distributes number_of_synapses on the model
        synaptic_locations = fill_synapses_vectors(HCell,number_of_synapses,rd,config)
        SynList = []
        ConList = []
        # Add human L2/L3 synapses on spine heads connected to synaptic_locations  
        add_synapses_on_list_of_segments(synaptic_locations,HCell,SynList,ConList,config)
        # To allow case where the synapses are activated a-synchronically
        configure_synaptic_delayes(MAX_JITTER,ConList,rd,config,cluster_type = None)
        # Run the simulation
        num_of_spikes,max_v = run_exp(HCell,config)
        spike_counts.append(num_of_spikes)
        max_v_list.append(max_v)

        # bar.update(jx*NUMBER_OF_RUNS_DIST+rep)

    spike_counts_distributed[number_of_synapses] = spike_counts
    max_v_distributed[number_of_synapses] = max_v

    


spike_counts_clusters = {}
max_v_clusters = {}
print("simulating the clustered case:")
# bar = progressbar.ProgressBar(max_value=len(CLUSTERS_RANGE*NUMBER_OF_RUNS_CLUSTER), widgets=[' [', progressbar.Timer(), '] ',
#         progressbar.Bar(), ' (', progressbar.ETA(), ') ',    ])

for jx,number_of_clusters in enumerate(CLUSTERS_RANGE):
    spike_counts = []
    max_v_list = []
    for rep in range(NUMBER_OF_RUNS_CLUSTER):
        # randomly distributes number_of_clusteres on the model
        synaptic_locations = fill_clustered_synapses_vectors(HCell,number_of_clusters,rd,config)
        SynList = []
        ConList = []
        # Add human L2/L3 synapses on spine heads connected to synaptic_locations  
        add_synapses_on_list_of_segments(synaptic_locations,HCell,SynList,ConList,config)
        # To allow case where the synapses are activated a-synchronically
        configure_synaptic_delayes(MAX_JITTER,ConList,rd,config,cluster_type = "sync_clusters")
        # Run the simulation
        num_of_spikes,max_v = run_exp(HCell,config)
        spike_counts.append(num_of_spikes)
        max_v_list.append(max_v)

        # bar.update(jx*NUMBER_OF_RUNS_CLUSTER+rep)

    spike_counts_clusters[number_of_clusters] = spike_counts
    max_v_clusters[number_of_clusters] = max_v_list

sp_prob_dist = []
sp_prob_dist_std = []
for n_s in DISTRIB_SYNS_RANGE:
    sp_prob_arr = (np.array(spike_counts_distributed[n_s])>0).astype(int)
    sp_prob_dist.append(np.mean(sp_prob_arr))
    sp_prob_dist_std.append(np.std(sp_prob_arr))

sp_prob_cluster = []
sp_prob_cluster_std = []
for n_s in CLUSTERS_RANGE:
    sp_prob_arr = (np.array(spike_counts_clusters[n_s])>0).astype(int)
    sp_prob_cluster.append(np.mean(sp_prob_arr))
    sp_prob_cluster_std.append(np.std(sp_prob_arr))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

lower_bound = [max(0,m-s) for (m,s) in zip(sp_prob_cluster,sp_prob_cluster_std)] 
higher_bound = [min(1,m+s) for (m,s) in zip(sp_prob_cluster,sp_prob_cluster_std)] 
ax1.fill_between(np.array(CLUSTERS_RANGE)*config.CLUSTER_SIZE, lower_bound, higher_bound, facecolor=COLOR_CLUSTER, alpha=0.5)

lower_bound = [max(0,m-s) for (m,s) in zip(sp_prob_dist,sp_prob_dist_std)] 
higher_bound = [min(1,m+s) for (m,s) in zip(sp_prob_dist,sp_prob_dist_std)] 
ax1.fill_between(DISTRIB_SYNS_RANGE, lower_bound, higher_bound, facecolor=COLOR_DISTRIBUTED, alpha=0.5)

ax1.plot(np.array(CLUSTERS_RANGE)*config.CLUSTER_SIZE,sp_prob_cluster,c=COLOR_CLUSTER,marker = 'o',ls='-',lw = 2)
ax1.plot(DISTRIB_SYNS_RANGE,sp_prob_dist,c=COLOR_DISTRIBUTED,marker = None,ls='-',lw = 2)


ax1.set_xticks(range(0,241,20))

ax1.set_xlim(0,240)
ax1.set_ylim(0,1)


ax1.set_ylabel('spike probability')
ax1.set_xlabel('number of synapses')

ax2.set_xticks(range(0,13,1))
ax2.set_xlabel("number of clusters")
