########
#
# This code generates the data for figure 3 in Eyal et al 2017
# It simulates 12,456 cases: 
# For each of the six human models in Eyal et al 2016 it goes over its electrical compartments 
# and simulates a synapse located on a human spine model connected to this compartment.
# For a comparison, the code also simulates activation of a synapse that was put directly on the shaft 
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

def create_model(model_file,morph_file,model_path = "../PassiveModels/",morph_path = "../morphs/"):

    # creating the model
    h.load_file("import3d.hoc")
    h.load_file("nrngui.hoc")
    h("objref cell, tobj")
    h.load_file(model_path+model_file+".hoc")
    h.execute("cell = new "+model_file+"()")
    nl = h.Import3d_Neurolucida3()
    nl.quiet = 1
    nl.input(morph_path+morph_file)
    imprt = h.Import3d_GUI(nl,0)
    imprt.instantiate(h.cell)
    HCell = h.cell
    HCell.geom_nseg()
    HCell.create_model()
    HCell.biophys()

    return HCell


# put an excitatory synapse on a spine head
def add_synapse_at_spine(HCell,num_of_synapses,sec_ref,x,SynList,ConList):
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

# put an excitatory synapse on directly on the shaft
def add_synapse_at_shaft(HCell,num_of_synapses,sec_ref,x,SynList,ConList):
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
def run_and_save_simulation(HCell,shaft_ref,x_shaft,f_output,on_spine = False,PLOT_RES=0):
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
    


def run_model(model,morph_file, path= "human_spines_simulations/",file_prefix = "simulating_spines_"):
    print model
    HCell = create_model(model,morph_file)
    output_file = path+file_prefix+model+".txt"
    f_output = open(output_file,'w')
    f_output.write("Spine,shaft_name,shaft_x,max_soma_v,max_shaft_v,max_spine_v,soma_v_integ,shaft_v_integ,spine_v_integ,ir_shaft,z_shaft,dist_shaft\n")

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
            SynList,ConList = add_synapse_at_spine(HCell,1,dend_sref,x,SynList,ConList)
            run_and_save_simulation(HCell,dend_sref,x,f_output,on_spine=True)
            f_output.write("%g,%g,%g\n"%(IR,IMP,dist))

            HCell.delete_spine()
            SynList = []
            ConList = []
            SynList,ConList = add_synapse_at_shaft(HCell,1,dend_sref,x,SynList,ConList)
            run_and_save_simulation(HCell,dend_sref,x,f_output,on_spine=False)
            f_output.write("%g,%g,%g\n"%(IR,IMP,dist))
        bar.update(jx)

    f_output.close()


run_model(model="cell0603_03_model_cm_0_49",morph_file="2013_03_06_cell03_789_H41_03.ASC")
run_model(model="cell0603_08_model_cm_0_45",morph_file="2013_03_06_cell08_876_H41_05_Cell2.ASC")
run_model(model="cell0603_11_model_cm_0_44",morph_file="2013_03_06_cell11_1125_H41_06.ASC")
run_model(model="cell1303_03_model_cm_0_43",morph_file="2013_03_13_cell03_1204_H42_02.ASC")
run_model(model="cell1303_05_model_cm_0_50",morph_file="2013_03_13_cell05_675_H42_04.ASC")
run_model(model="cell1303_06_model_cm_0_52",morph_file="2013_03_13_cell06_945_H42_05.ASC")

















