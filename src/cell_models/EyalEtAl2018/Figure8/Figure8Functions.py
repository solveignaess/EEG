
########
#
# Functions that are used in Figure 8. 
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########

import numpy as np
from neuron import h, gui
import matplotlib.pyplot as plt
import math


def random_synapse(HCell,rd,total_L, total_basal_L):
    """
    returns a random location in HCell - a neuron model
    rd -- NEURON random object
    total_L -- total dendritic length
    total_basal_L -- total basal length
    they are used here to choose a synaptic location out of the uniform distribution of dendritic locations
    that give the same probability to any point on the dendritic tree 
    note that just choosing segments randomly would ignore the segments physical length and would bias
    more synapses on shorter segments
    """

    synaptic_loc = rd.uniform(0,total_L)
    if synaptic_loc<total_basal_L:
        return basal_random_synapse(HCell,synaptic_loc)
    else:
        return apical_random_synapse(HCell,synaptic_loc-total_basal_L)


def basal_random_synapse(HCell,synaptic_loc):
    ''' returns a random location in the basal tree of this cell'''
    len0 = 0
    len1 = 0
    for sec in HCell.basal:
        len1 += sec.L
        if len1 >= synaptic_loc:
            x = (synaptic_loc-len0)/sec.L
            return sec,x
        h.pop_section()
        len0 = len1


def apical_random_synapse(HCell,synaptic_loc):
    ''' returns a random location in the apical tree of this cell'''
    len0 = 0
    len1 = 0
    for sec in HCell.apical:
        len1 += sec.L
        if len1 >= synaptic_loc:
            x = (synaptic_loc-len0)/sec.L
            return sec,x
        h.pop_section()
        len0 = len1

# Choose random number_of_synapses locations on HCell
# config is an object with the configurations required for this figure
def fill_synapses_vectors(HCell,number_of_synapses,rd,config):

    total_basal_L = sum([sec.L for sec in HCell.basal])
    total_L = sum([sec.L for sec in HCell.basal]) + sum([sec.L for sec in HCell.apical]) 
    syn_segments = []
    for i in range(number_of_synapses):
        sec,x = random_synapse(HCell,rd,total_L,total_basal_L)
        syn_segments.append(sec(x))

    return syn_segments

 

def fill_clustered_synapses_vectors(HCell,number_of_clusters,rd,config):
    '''
    Chooses random number_of_clusters locations on HCell
    The center of each cluster is chosen randomly, and the synapses within a cluster
    are distributed within config.CLUSTER_L which is 20 um in this work
    '''
    total_basal_L = sum([sec.L for sec in HCell.basal])
    total_L = sum([sec.L for sec in HCell.basal]) + sum([sec.L for sec in HCell.apical]) 

    syn_segments = []
    for i in range(number_of_clusters):
        sec,X_center = random_synapse(HCell,rd,total_L,total_basal_L)
        for i in range(config.CLUSTER_SIZE):
            if sec.L<config.CLUSTER_L:
                x = rd.uniform(0,1) 
            elif X_center<config.CLUSTER_L/sec.L:
                x = rd.uniform(0,config.CLUSTER_L/sec.L) 
            elif X_center>(1-config.CLUSTER_L/sec.L):
                x = rd.uniform(1-config.CLUSTER_L/sec.L,1)
            else: # the standard case
                x = rd.uniform(X_center-config.CLUSTER_L/2.0/sec.L,X_center+config.CLUSTER_L/2.0/sec.L)

            syn_segments.append(sec(x))

    return syn_segments

# The cluster locations shown in Figure 8C
def fill_clustered_synapses_demo(HCell,number_of_clusters,rd,config):
    segments_for_figure_7 = [(HCell.apic[38],0.85),(HCell.apic[69],0.5),(HCell.apic[90],0.7),
                            (HCell.dend[84],0.75),(HCell.dend[43],0.2),(HCell.dend[41],0.65)]
    syn_segments = []
    for i in range(number_of_clusters):
        sec,X_center = segments_for_figure_7[i]
        for i in range(config.CLUSTER_SIZE):
            if sec.L<config.CLUSTER_L:
                x = rd.uniform(0,1) 
            elif X_center<config.CLUSTER_L/sec.L:
                x = rd.uniform(0,config.CLUSTER_L/sec.L) 
            elif X_center>(1-config.CLUSTER_L/sec.L):
                x = rd.uniform(1-config.CLUSTER_L/sec.L,1)
            else: # the standard case
                x = rd.uniform(X_center-config.CLUSTER_L/2.0/sec.L,X_center+config.CLUSTER_L/2.0/sec.L)

            syn_segments.append(sec(x))

    return syn_segments


# Add spines to the input model, where the spines are connected to the segments in list_of_segments
def add_spines_on_segments(HCell,list_of_segments):

    Xs_vec = h.Vector()
    secs_sref_list = h.List()

    for seg in list_of_segments:
        sec = seg.sec
        sref = h.SectionRef(sec=sec)
        secs_sref_list.append(sref)
        Xs_vec.append(min(seg.x,1))

    HCell.add_few_spines(secs_sref_list,Xs_vec,0.25,1.35,2.8,HCell.soma[0].Ra)


# Add synapses on the top of human spine models that will be connected to the segments in list_of_segments
# The synaptic properties are defined in the config object and here are as in Figure 5.
def add_synapses_on_list_of_segments(list_of_segments,HCell,SynList,ConList,config):
    HCell.delete_spine()
    if len(list_of_segments) == 0:
        return

    add_spines_on_segments(HCell,list_of_segments)
    num_of_synapses = len(list_of_segments)
    for j in range(num_of_synapses):

            SynList.append(h.Exp2Syn(config.SPINE_HEAD_X,sec=HCell.spine[(j*2)+1]))
            ConList.append(h.NetCon(config.stim,SynList[-1]))
            SynList[-1].e = config.E_SYN
            SynList[-1].tau1 = config.TAU_1_AMPA
            SynList[-1].tau2 = config.TAU_2_AMPA
            ConList[-1].weight[0]= config.AMPA_W


    for j in range(num_of_synapses):
        
        SynList.append(h.NMDA(config.SPINE_HEAD_X,sec=HCell.spine[(j*2)+1]))
        ConList.append(h.NetCon(config.stim,SynList[-1]))
        SynList[-1].e=config.E_SYN
        SynList[-1].tau_r_NMDA = config.TAU_1_NMDA
        SynList[-1].tau_d_NMDA = config.TAU_2_NMDA
        SynList[-1].n_NMDA = config.N_NMDA
        SynList[-1].gama_NMDA = config.GAMMA_NMDA
        ConList[-1].weight[0] = config.NMDA_W




# Configure synaptic delays. 
# All figures in Eyal et al are based on sync activation, 
# But one may insert jittering as in Farinella et al., 2014
def configure_synaptic_delayes(jitter_time,ConList,rd,config,cluster_type = None):
    number_of_synapses = len(ConList)/2
    if jitter_time == 0: # no delay all the synapses are activated simultanouesly
        for i in range(len(ConList)):
            ConList[i].delay = jitter_time

    else:
        if cluster_type is None or cluster_type == "async_clusters" : #the synapses are activated a-synchronically
            for i in range(number_of_synapses):
                ConList[i].delay = rd.uniform(0,jitter_time)
                ConList[number_of_synapses+i].delay = ConList[i]

        if cluster_type == "sync_clusters": # synapses within a cluster are activated synchronically
            number_of_clusters = number_of_synapses/config.CLUSTER_SIZE
            for i in range(number_of_clusters):
                delay = rd.uniform(0,jitter_time)
                for j in range(config.CLUSTER_SIZE):
                    ConList[i*config.CLUSTER_SIZE+j].delay = delay
                    ConList[number_of_synapses+i*config.CLUSTER_SIZE+j].delay = delay
                    

# Runs a single experiment
# and returns the number of spikes in this run
# and the peak somatic voltage 
def run_exp(HCell,config):
    Vsoma = h.Vector()
    Vsoma.record(HCell.soma[0](.5)._ref_v)
    tvec = h.Vector()
    tvec.record(h._ref_t)
    apc = h.APCount(0.5,sec= HCell.soma[0])   
    apc.thresh = 0             
    apc.time   = 10000000.   

    h.init(h.v_init)
    h.run()

    np_t = np.array(tvec)
    np_v =  np.array(Vsoma)
    max_v = np.max(np_v[np.where(np_t>config.Spike_time)[0][0]:])
    return apc.n,max_v


 



