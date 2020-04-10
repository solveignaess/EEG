#!/usr/bin/python
########
#
# This code generates Figure 1D in Eyal 2017
# for each of the four experimental EPSP in Fig1D the code run a simulation 
# that puts a synapse with its fitted synaptic strength in the correct location.
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########
from neuron import h,gui
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pdb
import math
import progressbar
import glob
import peak_AMPA_cond_per_syn 

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
PATH_EPSP = 'ExpEPSP/Dat_Files/'
EPSP_SUFFIX = '.dat'
FIT_PREFIX_1_syn = 'fit_1_synapse/fit_1_syn_'
FIT_PREFIX_5_syn = 'fit_5_synapses/fit_5_syn_'

FIT_SUFFIX = '.txt'
TAU_1 = 0.3
TAU_2 = 1.8
E_SYN = 0
INIT_WEIGHT = 0.01
E_PAS = -86
DELAY = 0
init_Spike_time = 240
MAX_T = 400

T_OF_INTEREST = 150
FITS_WERE_RECENTLY_RAN = 1 # Change to 1 if you recently fitted the files.

# run the simulation
def run_exp(HCell,h):
    h.init(h.v_init)
    Vvec = h.Vector()
    Vvec.record(HCell.soma[0](0.5)._ref_v)
    h.run()
    np_v = np.array(Vvec)
    np_v = np_v[1:] # remove the 0 timing that is not part of the EPSPs data
    return np_v


def read_epsp_file(filename):
    f1 = open(filename)
    M = np.loadtxt(filename,skiprows=2,delimiter='\t')
    max_ix = int(MAX_T/(M[1,0]-M[0,0]))
    return M[0:max_ix,0],M[0:max_ix,1]



# This function reads the optimization result for the synaptic strength as described in Eyal et al.
def read_fit_file(filename,NUMBER_OF_SYNAPSES):
    f1 = open(filename)
    weights = []
    spike_times = []
    line =f1.readline() # title
    syns = []
    line =f1.readline()
    while line:
        L = line.split('\t')
        if len(L)<2: #last line
            continue
        d = {}
        d['trees'] = []
        d['secs'] = []
        d['segs'] = []

        for i in range(NUMBER_OF_SYNAPSES):
            d['trees'].append(L[2+i*3])
            d['secs'].append(int(L[3+i*3]))
            d['segs'].append(float(L[4+i*3]))
        syns.append(d)
        weights.append(float(L[-3]))
        spike_times.append(float(L[-2]))
        line =f1.readline()
    return syns,weights,spike_times


# This function receives as input an experimental EPSP and plot its voltage trace and the corresponding fits.
def plot_fit_syn(filename,c,fig,NUMBER_OF_SYNAPSES=1):

    plt.figure(fig)
    T_DATA,V_DATA = read_epsp_file(PATH_EPSP+filename+EPSP_SUFFIX)


    plt.plot(T_DATA,V_DATA,color='k')

    h.dt = T_DATA[1]-T_DATA[0]
    h.steps_per_ms = 1.0/h.dt
    h.tstop = T_DATA[-1]

    E_PAS = np.mean(V_DATA[0:int(100/h.dt)+1])
    for sec in HCell.all:
        sec.e_pas = E_PAS
    h.v_init = E_PAS

    if NUMBER_OF_SYNAPSES ==1:
        fit_prefix =  FIT_PREFIX_1_syn

    elif NUMBER_OF_SYNAPSES ==5:
        fit_prefix =  FIT_PREFIX_5_syn

        

    syns,weights,spike_times = read_fit_file(fit_prefix+filename+FIT_SUFFIX,NUMBER_OF_SYNAPSES)
    bar = progressbar.ProgressBar(max_value=len(syns), widgets=[' [', progressbar.Timer(), '] ',
                         progressbar.Bar(), ' (', progressbar.ETA(), ') ',    ])
    Stim1 = h.NetStim()
    Stim1.interval=10000 
    
    Stim1.noise=0
    Stim1.number=1
    

    for ix in range(len(syns)):
        Stim1.start=spike_times[ix]
        trees = syns[ix]['trees']
        secs = syns[ix]['secs']
        segs = syns[ix]['segs']
        Synlist = []
        Conlist = []
    
        for j in range(NUMBER_OF_SYNAPSES):
            if trees[j] == 'dend':
                sec = HCell.dend[secs[j]]
            else:
                sec = HCell.apic[secs[j]]
            seg = segs[j]
            Synlist.append(h.Exp2Syn(seg,sec=sec))

            Synlist[j].e=E_SYN
            Synlist[j].tau1=TAU_1
            Synlist[j].tau2=TAU_2
            Conlist.append(h.NetCon(Stim1,Synlist[j]))
            Conlist[j].weight[0] = weights[ix]
            
            Conlist[j].delay = DELAY
        model_v = run_exp(HCell,h)
        plt.plot(T_DATA,model_v,color = c,lw = 3)


        plt.xlim(T_DATA[np.argmax(model_v)]-20,T_DATA[np.argmax(model_v)]+40)
        plt.ylim(np.min(V_DATA)-0.1,np.min(V_DATA)+1.5)
        bar.update(ix)

# must run peak_AMPA_cond_per_syn before the plot of the fits
if FITS_WERE_RECENTLY_RAN:
    peak_AMPA_cond_per_syn.group_fits_1_syn(PATH_w_1_syn = "fit_1_synapse/")
    peak_AMPA_cond_per_syn.group_fits_5_syn(PATH_w_5_syn = "fit_5_synapses/")


print("Figure 1D1")
plot_fit_syn('081212_1to5',[1,0,0.22],1,NUMBER_OF_SYNAPSES=5)
print("Figure 1D2")
plot_fit_syn('110426_Sl4_Cl2_4to6',[1, 0.6 , 0.78],2,NUMBER_OF_SYNAPSES=5)
print("Figure 1D3")
plot_fit_syn('110426_Sl4_Cl2_6to4',[0, 0.5, 1],3,NUMBER_OF_SYNAPSES=5)
print("Figure 1D4")
plot_fit_syn('110322_Sl2_Cl2_6to4',[1 ,204.0/255.0, 0],4,NUMBER_OF_SYNAPSES=5)


    



