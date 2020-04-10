########
#
# This code generates figure S4 in Eyal et al 2017
# It recreates the fit of the model from Figure4 on five other recorded EPSPs
# The models were fitted as described in the manuscript
# Briefly, the model properties were constant as found for the Figure4 model
# and only the number of synapses and their locations were  

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


READ_FROM_PICKLE = 1

TAU_1_AMPA = 0.3
TAU_2_AMPA = 1.8
# properties of the model from figure 4:
TAU_1_NMDA = 8.019 
TAU_2_NMDA = 34.9884
N_NMDA = 0.28011
GAMMA_NMDA = 0.0765685
AMPA_W = 0.00073027
AMPA_W_B = 0.000130939
NMDA_W = 0.00131038

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



def plot_exp_epsps(cell,inj_ix=1,path = "../Figure4/data/",lw = 2):
    filename = path+cell+"_inj_loc_"+str(inj_ix)
    EPSP = np.loadtxt(filename+".txt")
    T = EPSP[10:,0]+10
    V = EPSP[10:,1] 
    plt.plot(T,V,'c',lw = lw)


   
    
# Add spines to the model
def add_spines(num_of_synapses,secs,Xs):
    secs_sref_list = h.List()
    Xs_vec = h.Vector()

    for ix in range(num_of_synapses): #create Neuron's list of section refs
        sref = h.SectionRef(sec = HCell.apic[secs[ix]])
        secs_sref_list.append(sref)
        Xs_vec.append(Xs[ix])

    HCell.add_few_spines(secs_sref_list,Xs_vec,0.25,1.35,2.8,HCell.soma[0].Ra)


# put an excitatory synapse on a spine head
def add_synapses_on_spines(num_of_synapses,SynList,ConList,model_properties,AMPA_BLOCKED = False):

    # Add AMPA synapses
    for j in range(num_of_synapses):
        SynList.append(h.Exp2Syn(SPINE_HEAD_X,sec=HCell.spine[(j*2)+1]))
        ConList.append(h.NetCon(Stim1,SynList[-1]))
        SynList[-1].e=E_SYN
        SynList[-1].tau1=TAU_1_AMPA
        SynList[-1].tau2=TAU_2_AMPA
        ConList[-1].delay = model_properties['DELAY']
        if AMPA_BLOCKED:
            ConList[-1].weight[0] = AMPA_W_B
        else:
            ConList[-1].weight[0] = AMPA_W

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
        ConList[-1].delay = model_properties['DELAY']



    return SynList,ConList


# run one simulation activation of synapses on spine heads) 
def run_and_save_simulation(c='r'):
    V_Soma = h.Vector()
    tvec = h.Vector()

    V_Soma.record(HCell.soma[0](0.5)._ref_v)
    tvec.record(h._ref_t)

    h.v_init = V_INIT
    h.init (h.v_init)
    h.run()

    plt.plot(np.array(tvec),np.array(V_Soma)-E_PAS,c=c)

# to show the location of the synapses (Fig 3B)
shp_list = []
def present_synapses(SynList,c=2,STYLE=4,SIZE=8):
    shp_list.append(h.Shape())
    shp_list[-1].rotate(0,0,0,0,0,5.25)

    shp_list[-1].show(0)
    for syn in SynList:
        shp_list[-1].point_mark(syn,c,STYLE,SIZE)

    shp_list[-1].view(-322,-600,620,1500,0+100*len(shp_list),0,200,450)


# The fit of the model to the other exp EPSPs is saved in two formats, pickle and csv
# The former is much easier, but if you don't have pickle change READ_FROM_PICKLE to 0
def read_csv():
    f = open("model_fits_to_other_EPSPs.csv")
    lines = f.readlines()
    f.close()
    model_syn_locations = {}

    for line in lines[1:]:
        
        L = line.strip().split('"')
        model = {}
        model['exp_EPSP'] = L[0].split(",")[1]
        model['inj_ix'] = int(L[0].split(",")[2])
        model['number_of_synapses'] = int(L[0].split(",")[3])
        model['DELAY'] = float(L[0].split(",")[4])
        model['rmsd'] = float(L[0].split(",")[5])
        model['syns_secs'] = [int(s) for s in L[1][1:-1].split(",")]
        model['syns_segs'] = [float(s) for s in L[3][1:-1].split(",")]

        model_syn_locations[(model['exp_EPSP'],model['inj_ix'])] = model

    return model_syn_locations



if READ_FROM_PICKLE:
    model_syn_locations = {}
    model_syn_locations_df = pnd.read_pickle('model_fits_to_other_EPSPs.p')
    for i in range(len(model_syn_locations_df)):
        model_syn_locations[(model_syn_locations_df.iloc[i]['exp_EPSP'],
            model_syn_locations_df.iloc[i]['inj_ix'])]=model_syn_locations_df.iloc[i]

else:
    model_syn_locations = read_csv()


# run over all 6 experimental EPSPs
# plot the exp EPSP
# read the model fit to this EPSP (each model fit are different location of the synapses)
# present the locations of the synapses using NEURON shape plot
# and run it + plot it
for jx,(exp_epsps,inj_ix) in enumerate(model_syn_locations):

    plt.figure(jx)

    plot_exp_epsps(exp_epsps,inj_ix)
    model_properties = model_syn_locations[(exp_epsps,inj_ix)]

    SynList = []
    ConList = []
    add_spines(model_properties['number_of_synapses'],model_properties['syns_secs'],model_properties['syns_segs'])
    SynList,ConList=add_synapses_on_spines(model_properties['number_of_synapses'],SynList,ConList,model_properties,AMPA_BLOCKED = False)
    run_and_save_simulation(c='r')


    plt.xlim(0,110)
    plt.ylim(-1,8)
    plt.tick_params(direction = 'in')

    # This will present the synapses as in the top of figure S4 on 6 different shp plots of NEURON
    present_synapses(SynList)















