########
#
# This code generates the data for figure 3 in Eyal et al 2017
# it simulates activation of synapses on spine models for human model cell0603_08.
# For a comparison the case of activating a synapse directly on the shaft is also simulated.
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########




from neuron import h,gui
import numpy as np
import matplotlib.pyplot as plt
import pandas as pnd
import matplotlib
import progressbar


# creating the model
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
FREQ = 100 # for impedance calculations

h.tstop = 100
V_INIT = E_PAS


Stim1 = h.NetStim()
Stim1.interval=10000 
Stim1.start=Spike_time
Stim1.noise=0
Stim1.number=1


output_file = "synapses_on_spines_cell0603_08.txt"
f_output = open(output_file,'w')
f_output.write("Spine,shaft_name,shaft_x,max_soma_v,max_shaft_v,max_spine_v,soma_v_integ,shaft_v_integ,spine_v_integ,ir_shaft,z_shaft,dist_shaft\n")

# put an excitatory synapse on a spine head
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

# put an excitatory synapse directly on the shaft
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

# run one simulation (activation of a synapse on a spine or on the shaft and save its result to the outputfile )
def run_and_save_simulation(shaft_ref,x_shaft,on_spine = False,PLOT_RES=0):
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
        plt.plot(np.array(tvec),np.array(V_Soma),c='g')
        plt.plot(np.array(tvec),np.array(V_Shaft),c='r')
        if on_spine:
            plt.plot(np.array(tvec),np.array(V_Spine),c='m')

    if on_spine:
        f_output.write("1,")
    else:
        f_output.write("0,")

    f_output.write("%s,%0.3f,"%(shaft_ref.sec.name(),x_shaft))
    if on_spine:

        f_output.write("%g,%g,%g,"%(V_Soma.max(),V_Shaft.max(),V_Spine.max()))
        f_output.write("%g,%g,%g,"%(V_Soma.sub(V_INIT).sum(),V_Shaft.sub(V_INIT).sum(),V_Spine.sub(V_INIT).sum()))
    else:
        f_output.write("%g,%g,%g,"%(V_Soma.max(),V_Shaft.max(),0))
        f_output.write("%g,%g,%g,"%(V_Soma.sub(V_INIT).sum(),V_Shaft.sub(V_INIT).sum(),0))
    

# run over all the compartments in the model
h.distance(0,0.5,sec=HCell.soma[0])
num_secs = len([sec for sec in HCell.all ])
bar = progressbar.ProgressBar(max_value=num_secs, widgets=[' [', progressbar.Timer(), '] ',
            progressbar.Bar(), ' (', progressbar.ETA(), ') ',    ])
for jx,sec in enumerate(HCell.all):

    dend_sref = h.SectionRef(sec = sec)

    Xs = [seg.x for seg in sec]
    Xs = Xs + [1]
    
    for x in Xs:
        h.finitialize(V_INIT)
        dist = h.distance(x,sec=sec)
        z = h.Impedance()
        z.loc(x,sec=sec)
        z.compute(0,1)
        IR = z.input(x,sec=sec)
        z.compute(FREQ,1)
        IMP = z.input(x,sec=sec)
        SynList = []
        ConList = []
        HCell.delete_spine() # delete the previous spines
        SynList,ConList = add_synapse_at_spine(1,dend_sref,x,SynList,ConList)
        run_and_save_simulation(dend_sref,x,on_spine=True)
        f_output.write("%g,%g,%g\n"%(IR,IMP,dist))
        
        SynList = []
        ConList = []

        SynList,ConList = add_synapse_at_shaft(1,dend_sref,x,SynList,ConList)
        run_and_save_simulation(dend_sref,x,on_spine=False)
        f_output.write("%g,%g,%g\n"%(IR,IMP,dist))
    bar.update(jx)



f_output.close()




















