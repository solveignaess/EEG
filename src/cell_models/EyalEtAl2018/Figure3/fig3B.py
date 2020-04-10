########
#
# This code generates Figure 3B in Eyal 2017
# It simulates a synapse on a the head of a spine model connected to basal dendrite
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########

from neuron import h,gui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

h.load_file("import3d.hoc")
h.load_file("nrngui.hoc")
h("objref cell, tobj")
morph_file = "../morphs/2013_03_06_cell08_876_H41_05_Cell2.ASC"
model_file = "cell0603_08_model_cm_0_45"
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

PLOT_MODE = 0

TAU_1 = 0.3
TAU_2 = 1.8
E_SYN = 0
WEIGHT = 0.00088 # from figure 1
E_PAS = -86
Spike_time = 10
DELAY = 0
NUM_OF_SYNAPSES = 1
SPINE_HEAD_X = 1

h.tstop = 100
V_INIT = E_PAS


Stim1 = h.NetStim()
Stim1.interval=10000 
Stim1.start=Spike_time
Stim1.noise=0
Stim1.number=1

DEND_example = 62
X_example = 0.5

c_spine = [181.0/256,170.0/256,31.0/256]
c_shaft = [0.0,0.5,0.0]
c_soma = [0.0, 174.0/256,239.0/256]

def add_synapse_at_spine(num_of_synapses,sec_ref,x,SynList,ConList):
    HCell.add_spine_ra(sec_ref,x,0.25,1.35,2.8,203.23)

    for j in range(num_of_synapses):
        SynList.append(h.Exp2Syn(SPINE_HEAD_X,sec=HCell.spine[(j*2)+1]))
        ConList.append(h.NetCon(Stim1,SynList[-1]))
        SynList[-1].e=E_SYN
        SynList[-1].tau1=TAU_1
        SynList[-1].tau2=TAU_2
        ConList[-1].weight[0] = WEIGHT
        ConList[-1].delay = DELAY

    return SynList,ConList

def add_synapse_at_shaft(num_of_synapses,sec_ref,x,SynList,ConList):
    SynList = []
    ConList = []
    for j in range(num_of_synapses):
        SynList.append(h.Exp2Syn(x,sec=sec_ref.sec))
        ConList.append(h.NetCon(Stim1,SynList[-1]))
        SynList[-1].e=E_SYN
        SynList[-1].tau1=TAU_1
        SynList[-1].tau2=TAU_2
        ConList[-1].weight[0] = WEIGHT
        ConList[-1].delay = DELAY

    return SynList,ConList


def run_and_save_simulation(shaft_ref,x_shaft,on_spine = False,PLOT_RES=1):
    V_Shaft = h.Vector()
    V_Soma = h.Vector()
    tvec = h.Vector()

    V_Soma.record(HCell.soma[0](0.5)._ref_v)
    V_Shaft.record(shaft_ref.sec(x_shaft)._ref_v)
    tvec.record(h._ref_t)

    if on_spine:
        V_Spine = h.Vector()
        V_Spine.record(HCell.spine[1](1)._ref_v)

    h.v_init = V_INIT
    h.init (h.v_init)
    h.run()

    if (PLOT_RES>0):
        if on_spine:
            plt.close('all')
        plt.plot(np.array(tvec),np.array(V_Soma),c=c_soma)
        plt.plot(np.array(tvec),np.array(V_Shaft),c=c_shaft)
        if on_spine:
            plt.plot(np.array(tvec),np.array(V_Spine),c=c_spine)

        plt.xlim(0,50)

    M = np.array([np.array(tvec),np.array(V_Soma),np.array(V_Shaft),np.array(V_Spine)]).T

    np.savetxt("example_stimulation of_a_spine_connected_to_dend_62.txt",M)


    


h.distance(0,0.5,sec=HCell.soma[0])

sec=HCell.dend[DEND_example]
x = X_example

dend_sref = h.SectionRef(sec = sec)

print dend_sref.sec.name()


h.finitialize(V_INIT)
dist = h.distance(x,sec=sec)
z = h.Impedance()
z.loc(x,sec=sec)
z.compute(0,1)

SynList = []
ConList = []
SynList,ConList = add_synapse_at_spine(1,dend_sref,x,SynList,ConList)
run_and_save_simulation(dend_sref,x,on_spine=True,PLOT_RES=1)
        























