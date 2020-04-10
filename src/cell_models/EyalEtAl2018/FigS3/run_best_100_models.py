########
#
# This code generates the data for figure S3 in Eyal et al 2017
# It reads the models from the folder of Figure S3 
# and then for each for one of the 100 AMPA+NMDA models,
# distributes 1-50 synapses on a basal dendrite of cell 130305 
# and measures the peak voltage as function of the number of synapses
# The run may yake about an hour.
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
import pickle
import progressbar


import os

os.system('nrnivmodl ../mechanisms/')

# creating the HL2/L3 model
h.load_file("import3d.hoc")
h.load_file("nrngui.hoc")
h("objref cell, tobj")
morph_file = "../morphs/2013_03_13_cell05_675_H42_04.ASC" # This is the cell the data in Fig3B was recorded from
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
def add_synapses_on_spines(num_of_synapses,SynList,ConList,model_properties):

    # Add AMPA synapses
    for j in range(num_of_synapses):
        SynList.append(h.Exp2Syn(SPINE_HEAD_X,sec=HCell.spine[(j*2)+1]))
        ConList.append(h.NetCon(Stim1,SynList[-1]))
        SynList[-1].e=E_SYN
        SynList[-1].tau1=TAU_1_AMPA
        SynList[-1].tau2=TAU_2_AMPA

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

        ConList[-1].delay = model_properties['DELAY']


    return SynList,ConList


# run one simulation 
def run_simulation():
    V_Soma = h.Vector()
    tvec = h.Vector()

    V_Soma.record(HCell.soma[0](0.5)._ref_v)
    tvec.record(h._ref_t)

    h.v_init = V_INIT
    h.init (h.v_init)
    h.run()

    return np.array(V_Soma)


# The top 100 models are saved in two formats, pickle and CSV. 
# The former is much easier, but if you don't have pickle change READ_FROM_PICKLE to 0
def read_csv():
    f = open("../Figure4/best_100_models.txt")
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
    models_df = pnd.read_pickle('../Figure4/best_100_models.p')
    d = models_df.to_dict(orient='list')
    models = {}
    for ix,model in enumerate(d['model']):
        models[model] = {}
        for k in d:
            models[model][k] = d[k][ix]


else:
    models = read_csv()

branch_L = HCell.dend[DEND_SEC].L
seg_Xs = DEND_X+(np.random.rand(1,50)*CLUSTER_LENGTH-CLUSTER_LENGTH/2)/branch_L    
seg_Xs = seg_Xs[0]
secs_list = [DEND_SEC]*50
trees_list = ['dend']*50

models_soma_peak_v = {}

f = open("models_num_syn_vs_v.txt",'w')
f.write("model," + ",".join([str(i) for i in range(1,51)])+"\n")

print "simulating the 100 models"
bar = progressbar.ProgressBar(max_value=len(models), widgets=[' [', progressbar.Timer(), '] ',
            progressbar.Bar(), ' (', progressbar.ETA(), ') ',    ])

for ix,(model_ix,model_properties) in enumerate(models.items()):
    models_soma_peak_v[model_ix] = []
    for num_of_synapses in range(1,51):
        SynList = []
        ConList = []
        add_spines(num_of_synapses,trees_list[:num_of_synapses],secs_list[:num_of_synapses],seg_Xs[:num_of_synapses])
        SynList,ConList = add_synapses_on_spines(num_of_synapses,SynList,ConList,model_properties)
        v_soma = run_simulation()
        models_soma_peak_v[model_ix].append(np.max(v_soma)-E_PAS)

    f.write(str(model_ix)+","+",".join("%g"%v for v in models_soma_peak_v[model_ix])+"\n")    
    bar.update(ix)


f.close()



