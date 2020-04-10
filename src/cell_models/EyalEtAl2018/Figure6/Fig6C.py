######
#
# This code creates Figure 6C in Eyal et al 2017
# It generates NMDA spike in 28 dendritic terminals of human model 130306
# Then it tests whether these NMDA spikes are independent according to the definition defined in Eyal et al
# The code presents the voltage 15 ms after the activation of the NMDA spikes
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########


import os

os.system('nrnivmodl ../mechanisms/')

from neuron import h,gui
import numpy as np
import matplotlib.pyplot as plt
import pandas as pnd
import matplotlib
import pickle


from tree_functions import *


h.load_file("import3d.hoc")
h.load_file("nrngui.hoc")


h("objref cell, tobj")
morph_file = "../morphs/2013_03_13_cell06_945_H42_05.ASC"
model_file = "cell1303_06_model_cm_0_52" # the model for cell 130306 as fitted in Eyal et al 2016
model_path = "../PassiveModels/"



h.load_file(model_path+model_file+".hoc")
h.execute("cell = new "+model_file+"()") #replace?
nl = h.Import3d_Neurolucida3()
nl.quiet = 1
nl.input(morph_file)
imprt = h.Import3d_GUI(nl,0)
imprt.instantiate(h.cell)
HCell = h.cell
HCell.geom_nseg()
HCell.create_model()
HCell.biophys()

# this file reads a database that maps from each segment in the model to the number of synapses
# required to genrate NMDA spike in this segment
# see more details in folder number_of_synapses_to_NMDA_spike 
dict_syn_to_nmda_spike_file = "../number_of_synapses_to_NMDA_spike/dict_1303_06_seg_to_nmda_spike.pickle"


TAU_1_AMPA = 0.3
TAU_2_AMPA = 1.8
# properties of the model from figure 4:
TAU_1_NMDA = 8.019 
TAU_2_NMDA = 34.9884
N_NMDA = 0.28011
GAMMA_NMDA = 0.0765685
AMPA_W = 0.00073027
NMDA_W = 0.00131038
NMDA_W_BLOCKED = 0


E_SYN = 0
E_PAS = -86
Spike_time = 10
DELAY = 0
NUM_OF_SYNAPSES = 1
SPINE_HEAD_X = 1
SECTION_TERMINAL = 1

h.tstop = 100
V_INIT = E_PAS

CLUSTER_LENGTH = 20

VOLTAGE_THRESHOLD_NEAREST_JUNCTION = -40


Stim1 = h.NetStim()
Stim1.interval=10000 
Stim1.start=Spike_time
Stim1.noise=0
Stim1.number=1

# for the plot shape
ps_rotation = [0,0,0,0,0,4.6]
ps_view = [-300,-300,600,1500,200,0,300,750]


# this function creates a dictionary that map from segment pointer to the number of synapses required 
# to generate NMDA spike in this segment
def seg_to_num_of_syn_per_nmda_spike_func(dict_filename,cell):
    with open(dict_syn_to_nmda_spike_file, 'rb') as handle:
        d = pickle.load(handle)

    seg_to_nmda_threshold = {}
    for sec in list(cell.basal)+list(cell.apical):
        for seg in list(sec)+[sec(1)]:
            if seg.x == 0:
                continue
            dot_index = str.rfind(sec.hname(),".")
            secname = sec.hname()[dot_index+1:]
            seg_to_nmda_threshold[seg]  = d[(secname,round(seg.x,3))]

    return seg_to_nmda_threshold


# Add spines to the model
def add_spines_on_segments(list_of_segments,seg_to_num_of_syn):
    HCell.delete_spine()
    total_synapses = 0

    Xs_vec = h.Vector()
    secs_sref_list = h.List()

    for seg in list_of_segments:
        sec = seg.sec
        num_of_synapses = seg_to_num_of_syn[seg]
        assert num_of_synapses!=0, "segment on %s has 0 synapses"%sec.hname()
        Lsec = sec.L
        sref = h.SectionRef(sec=sec)

        if Lsec > CLUSTER_LENGTH: # distributes the spines on CLUSTER_LENGTH um on the section
            min_x = (Lsec-CLUSTER_LENGTH)/float(seg.x)
        else: # in cases where the section is shorter than CLUSTER_LENGTH
            mix_x = 0

        for ix in range(num_of_synapses):
            secs_sref_list.append(sref)
            x_syn = min_x+ix*CLUSTER_LENGTH/float(num_of_synapses-1)
            Xs_vec.append(min(x_syn/float(Lsec),1))

        total_synapses += num_of_synapses

    HCell.add_few_spines(secs_sref_list,Xs_vec,0.25,1.35,2.8,HCell.soma[0].Ra)

    HCell.soma[0].push()
    return total_synapses




# put excitatory synapses on spine heads 
# list_of_segments defines the locations
# seg_to_num_of_syn defines the number of synapses to put in each location
def add_synapses_on_list_of_segments(list_of_segments,SynList,ConList,seg_to_num_of_syn):
    # delete previous synapses
    HCell.delete_spine()

    num_of_synapses = add_spines_on_segments(list_of_segments,seg_to_num_of_syn)

    for j in range(num_of_synapses):

            SynList.append(h.Exp2Syn(SPINE_HEAD_X,sec=HCell.spine[(j*2)+1]))
            ConList.append(h.NetCon(Stim1,SynList[-1]))
            SynList[-1].e=E_SYN
            SynList[-1].tau1=TAU_1_AMPA
            SynList[-1].tau2=TAU_2_AMPA
            ConList[-1].weight[0]= AMPA_W
            ConList[-1].delay = DELAY


    for j in range(num_of_synapses):
        
        SynList.append(h.NMDA(SPINE_HEAD_X,sec=HCell.spine[(j*2)+1]))
        ConList.append(h.NetCon(Stim1,SynList[-1]))
        SynList[-1].e=E_SYN
        SynList[-1].tau_r_NMDA=TAU_1_NMDA
        SynList[-1].tau_d_NMDA=TAU_2_NMDA
        ConList[-1].weight[0] = NMDA_W
        ConList[-1].delay = DELAY
        SynList[-1].n_NMDA = N_NMDA
        SynList[-1].gama_NMDA = GAMMA_NMDA



    return SynList,ConList

# This function tests whether NMDA spike is "independent"
# independency of a spike (subunit) is defined if the voltage in the most proximal junction between 
# this terminal and any other activated terminal has never passed the voltage in VOLTAGE_THRESHOLD_NEAREST_JUNCTION
# see more details in Eyal et al.
def test_independance_of_NMDA_spikes(list_of_terminals,sec_voltage_dict):

    success = 1
    for ix,seg1 in enumerate(list_of_terminals):
        sec1 = seg1.sec
        for jx,seg2 in enumerate(list_of_terminals[ix+1:]):
            sec2 = seg2.sec
            junction = nearest_junction(sec1,sec2,HCell.soma[0])
            if np.max(np.array(sec_voltage_dict[junction]))>VOLTAGE_THRESHOLD_NEAREST_JUNCTION:
                print "fail on:"
                print sec1.hname()
                print sec2.hname()
                print 
                success = 0

    return success


def create_plot_shape(rotation,view):

    h.load_file("../NEURON_color_maps/TColorMap.hoc")
    h.load_file("movierun.hoc")

    ps = h.PlotShape() 
    ps.exec_menu("View = plot")
    ps.variable("v")
    cm1 = h.TColorMap("../NEURON_color_maps/jet.cm")


    cm1.set_color_map(ps,-90,-10)
    h.fast_flush_list.append(ps)
    ps.exec_menu("Shape Plot")
    ps.exec_menu("Show Diam")


    ps.exec_menu("Variable Scale")
    try:
        ps.rotate(rotation[0],rotation[1],rotation[2],rotation[3],rotation[4],rotation[5])

        ps.view(view[0],view[1],view[2],view[3],view[4],view[5],view[6],view[7])
    except:
        import pdb
        pdb.set_trace()
    ps.exec_menu("View Box")


SynList = []
ConList = []
sec_voltage_dict = {}


# Tests the list of input terminals. If all of them are independent 
# generates plot shape and display the voltage plot_ms after the activation of the NMDA spikes
def run_exp(basal_list,apic_list,seg_to_num_of_syn,plot_ms = 15):
    nmda_tips = []
    for sec_ix in apic_list:
        nmda_tips.append(HCell.apic[sec_ix](SECTION_TERMINAL))

    for sec_ix in basal_list:
        nmda_tips.append(HCell.dend[sec_ix](SECTION_TERMINAL))

    del SynList[:]
    del ConList[:]

    add_synapses_on_list_of_segments(nmda_tips,SynList,ConList,seg_to_num_of_syn)


    # Record voltage in all the sections of the model
    sec_voltage_dict.clear()
    for sec in [HCell.soma[0]]+list(HCell.basal)+list(HCell.apical):
        sec_voltage_dict[sec] = h.Vector()

    sec_voltage_dict[HCell.soma[0]].record(HCell.soma[0](.5)._ref_v)
    for sec in list(HCell.basal)+list(HCell.apical):
        sec_voltage_dict[sec].record(sec(1)._ref_v)
    
    h.v_init= V_INIT
    h.init (h.v_init)
    h.run()

    success = test_independance_of_NMDA_spikes(nmda_tips,sec_voltage_dict)
    if success:
        create_plot_shape(ps_rotation,ps_view)
        h.tstop = Spike_time+plot_ms
        h.run()

    else:
        print "Failure: The NMDA spikes are not independent"

    return success


# List of sections with NMDA tips in their head
apic_list = [21,26,29,37,38,39,42,45,50,54,65,70]
basal_list = [6,12,14,18,22,25,33,40,46,49,54,57,61,64,73,76]

seg_to_num_syn_for_nmda_spike = seg_to_num_of_syn_per_nmda_spike_func(dict_syn_to_nmda_spike_file,HCell)
success = run_exp(basal_list,apic_list,seg_to_num_syn_for_nmda_spike)

