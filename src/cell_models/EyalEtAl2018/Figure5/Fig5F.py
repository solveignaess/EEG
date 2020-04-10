########
#
# This code generates figure 5F in Eyal et al 2017
# It activates an NMDA spike on one of the basal trees of human model 130305. 
# The NMDA model here is the same as fitted in Figure 4
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



# creating the model
h.load_file("import3d.hoc")
h.load_file("nrngui.hoc")
h("objref cell, tobj")
morph_file = "../morphs/2013_03_13_cell05_675_H42_04.ASC" # This is the cell the data in Fig4B was recorded from
model_file = "cell1303_05_model_cm_0_50" # the model for cell 130305 as fitted in Eyal et al 2016
model_path = "../PassiveModels/"
h.load_file(model_path+model_file+".hoc")
h.execute("cell = new "+model_file+"()")
nl = h.Import3d_Neurolucida3()
nl.quiet = 1
nl.input(morph_file)
imprt = h.Import3d_GUI(nl,0)
imprt.instantiate(h.cell)
HCell = h.cell
HCell.geom_nseg()
HCell.create_model()
HCell.biophys()


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

h.tstop = 250
V_INIT = E_PAS


Stim1 = h.NetStim()
Stim1.interval=10000 
Stim1.start=Spike_time
Stim1.noise=0
Stim1.number=1

DEND_SEC = 65
DEND_X = 0.5

CLUSTER_LENGTH = 20

# Add spines to the model
def add_spines(num_of_synapses,trees,secs,Xs):
    HCell.delete_spine()
    secs_sref_list = h.List()
    Xs_vec = h.Vector()

    for ix in range(num_of_synapses): #create Neuron's list of section refs
        if trees[ix] == 'apic':
            sref = h.SectionRef(sec = HCell.apic[secs[ix]])
        else:
            sref = h.SectionRef(sec = HCell.dend[secs[ix]])
        secs_sref_list.append(sref)

        Xs_vec.append(Xs[ix])

    HCell.add_few_spines(secs_sref_list,Xs_vec,0.25,1.35,2.8,HCell.soma[0].Ra)


# put excitatory synapses on spine heads
def add_synapses_on_spines(num_of_synapses,SynList,ConList,NMDA_BLOCKED = False):

    # Add AMPA synapses
    for j in range(num_of_synapses):
        SynList.append(h.Exp2Syn(SPINE_HEAD_X,sec=HCell.spine[(j*2)+1]))
        ConList.append(h.NetCon(Stim1,SynList[-1]))
        SynList[-1].e=E_SYN
        SynList[-1].tau1=TAU_1_AMPA
        SynList[-1].tau2=TAU_2_AMPA

        ConList[-1].weight[0] = AMPA_W
        ConList[-1].delay = DELAY

    # Add NMDA synapses
    for j in range(num_of_synapses):
        SynList.append(h.NMDA(SPINE_HEAD_X,sec=HCell.spine[(j*2)+1]))
        ConList.append(h.NetCon(Stim1,SynList[-1]))
        SynList[-1].e=E_SYN
        SynList[-1].tau_r_NMDA = TAU_1_NMDA
        SynList[-1].tau_d_NMDA = TAU_2_NMDA
        SynList[-1].n_NMDA = N_NMDA
        SynList[-1].gama_NMDA = GAMMA_NMDA
        ConList[-1].weight[0] = NMDA_W
        if NMDA_BLOCKED:
            ConList[-1].weight[0] = NMDA_W_BLOCKED

        ConList[-1].delay = DELAY


    return SynList,ConList


# run one simulation 
def run_simulation():
    V_Soma = h.Vector()
    V_Dend = h.Vector()
    tvec = h.Vector()

    V_Soma.record(HCell.soma[0](0.5)._ref_v)
    V_Dend.record(HCell.dend[DEND_SEC](DEND_X)._ref_v)
    tvec.record(h._ref_t)

    h.v_init = V_INIT
    h.init (h.v_init)
    h.run()

    return np.array(tvec),np.array(V_Soma),np.array(V_Dend)




branch_L = HCell.dend[DEND_SEC].L
seg_Xs = DEND_X+(np.random.rand(1,50)*CLUSTER_LENGTH-CLUSTER_LENGTH/2)/branch_L    
seg_Xs = seg_Xs[0]
secs_list = [DEND_SEC]*30
trees_list = ['dend']*30


h.load_file("../NEURON_color_maps/TColorMap.hoc")
h.load_file("movierun.hoc")

ps = h.PlotShape() 
ps.exec_menu("View = plot")
ps.variable("v")
cm1 = h.TColorMap("../NEURON_color_maps/jet.cm")
ps.rotate(0,0,0,0,0,5.25)
cm1.set_color_map(ps,-90,-10)

ps.exec_menu("Shape Plot")
ps.exec_menu("Show Diam")
ps.exec_menu("Variable Scale")


ps.view(-322,-600,620,1500,800,0,800,1800)

SYNPASES_FOR_NMDA_SPIKE = 20
TSTOP = 10
h.tstop = Spike_time+TSTOP

num_of_synapses = SYNPASES_FOR_NMDA_SPIKE
SynList = []
ConList = []
add_spines(num_of_synapses,trees_list[:num_of_synapses],secs_list[:num_of_synapses],seg_Xs[:num_of_synapses])
SynList,ConList = add_synapses_on_spines(num_of_synapses,SynList,ConList)
tvec,v_soma,v_dend = run_simulation()


