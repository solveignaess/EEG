########
#
# This code generates figure 4 in Eyal et al 2017
# It first reads the data files and displays them
# Then it reads the top 100 models and simulates one of them
# The models were fitted as described in the manuscript
# Each model has its NMDA conductance and kinetics 
# as well as AMPA conductance and AMPA conductance during the AMPA block

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

# This is the model used in the paper.
# choose any other of the top 100 models in "best_100_models.txt" to see their fits
MODEL_IX = 15796 

READ_FROM_PICKLE = 1

TAU_1_AMPA = 0.3
TAU_2_AMPA = 1.8
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



def plot_exp_epsps(cell,inj_ix=1,path = "data/",lw = 1):
    filename = path+cell+"_inj_loc_"+str(inj_ix)
    EPSP = np.loadtxt(filename+".txt")
    T = EPSP[10:,0]+10
    V = EPSP[10:,1] + E_PAS
    plt.plot(T,V,'c',lw = lw)

    filename+="_with_blockers"
    B_EPSP = np.loadtxt(filename+".txt")
    T = B_EPSP[10:,0]+10
    V = B_EPSP[10:,1] + E_PAS
    plt.plot(T,V,'k',lw = lw)

    plt.xlim(0,210)
    plt.ylim(-87,-78)

   
    
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
        if AMPA_BLOCKED:
            ConList[-1].weight[0] = model_properties['AMPA_W_B']
            ConList[-1].delay = model_properties['DELAY_B']
        else:
            ConList[-1].weight[0] = model_properties['AMPA_W']
            ConList[-1].delay = model_properties['DELAY']

    # Add NMDA synapses
    for j in range(num_of_synapses):
        SynList.append(h.NMDA(SPINE_HEAD_X,sec=HCell.spine[(j*2)+1]))
        ConList.append(h.NetCon(Stim1,SynList[-1]))
        SynList[-1].e=E_SYN
        SynList[-1].tau_r_NMDA = model_properties['tau_1_NMDA']
        SynList[-1].tau_d_NMDA = model_properties['tau_2_NMDA']
        SynList[-1].n_NMDA = model_properties['N_NMDA']
        SynList[-1].gama_NMDA = model_properties['GAMMA_NMDA']
        ConList[-1].weight[0] = model_properties['NMDA_W']
        if AMPA_BLOCKED:
            ConList[-1].delay = model_properties['DELAY_B']
        else:
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

    plt.plot(np.array(tvec),np.array(V_Soma),c=c)

# to show the location of the synapses (Fig 4B)
def present_synapses(SynList,c=2,STYLE=4,SIZE=8):
    shp = h.Shape()
    shp.rotate(0,0,0,0,0,5.25)

    shp.show(0)
    for syn in SynList:
        shp.point_mark(syn,c,STYLE,SIZE)

    shp.view(-322,-600,620,1500,800,0,800,1800)


# The top 100 models are saved in two formats, pickle and CSV. 
# The former is much easier, but if you don't have pickle change READ_FROM_PICKLE to 0
def read_csv():
    f = open("best_100_models.txt")
    lines = f.readlines()
    f.close()
    models = {}

    for line in lines[1:]:
        
        L = line.strip().split('"')
        model = {}
        model['model'] = int(L[0].split(",")[1])
        model['number_of_synapses'] = int(L[0].split(",")[2])
        model['NMDA_W'] = float(L[0].split(",")[3])
        model['tau_1_NMDA'] = float(L[0].split(",")[4])
        model['tau_2_NMDA'] = float(L[0].split(",")[5])
        model['N_NMDA'] = float(L[0].split(",")[6])
        model['GAMMA_NMDA'] = float(L[0].split(",")[7])
        model['AMPA_W'] = float(L[0].split(",")[8])
        model['DELAY'] = float(L[0].split(",")[9])
        model['rmsd_epsp'] = float(L[0].split(",")[10])
        model['AMPA_W_B'] = float(L[0].split(",")[11])
        model['DELAY_B'] = float(L[0].split(",")[12])
        model['rmsd_block'] = float(L[0].split(",")[13])
        model['syns_secs'] = [int(s) for s in L[1][1:-1].split(",")]
        model['syns_segs'] = [float(s) for s in L[3][1:-1].split(",")]

        models[model['model']] = model

    return models



if READ_FROM_PICKLE:
    models = pnd.read_pickle('best_100_models.p')
    model_properties = models[models['model']==MODEL_IX]
    model_properties = model_properties.to_dict(orient='records')[0]
else:
    models = read_csv()
    model_properties = models[MODEL_IX]

# Display the three experimental examples in figure 3A
plt.figure(1)
plot_exp_epsps("2013_03_13Cel05",inj_ix=1)
plt.figure(2)
plot_exp_epsps("2013_03_20Cel04",inj_ix=1)
plt.figure(3)
plot_exp_epsps("2013_03_20Cel06",inj_ix=2)


# Run the simulation with the fitted model to the experimental case in Fig4A1
plt.figure(4)
plot_exp_epsps("2013_03_13Cel05",inj_ix=1,lw=2)

SynList = []
ConList = []
add_spines(model_properties['number_of_synapses'],model_properties['syns_secs'],model_properties['syns_segs'])
SynList,ConList=add_synapses_on_spines(model_properties['number_of_synapses'],SynList,ConList,model_properties,AMPA_BLOCKED = False)
run_and_save_simulation(c='r')

# Run the same model but with blocked AMPA conductance
SynList = []
ConList = []
SynList,ConList=add_synapses_on_spines(model_properties['number_of_synapses'],SynList,ConList,model_properties,AMPA_BLOCKED = True)
run_and_save_simulation(c='r')

plt.xlim(0,110)
plt.ylim(-87,-78)
present_synapses(SynList)














