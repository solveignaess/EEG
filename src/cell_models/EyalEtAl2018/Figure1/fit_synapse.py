########
#
# This code fits the synaptic conductance of a one synapse to fit one experimental EPSP
# The synaptic location and the name of the experimental EPSPs should be given as input to the script.
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########
from neuron import h,gui
import sys,os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pnd
import matplotlib
import math



try:
    sys.path.append("/home/ls/users/guy.eyal/Passive_Paper/codeForFrontiersPaper")
except:
    pass

def read_epsp_file(filename):
    f1 = open(filename)
    M = np.loadtxt(filename,skiprows=2,delimiter='\t')
    max_ix = int(MAX_T/(M[1,0]-M[0,0]))
    f1.close()
    return M[0:max_ix,0],M[0:max_ix,1]


def rms(predictions,targets):
    n = len(predictions)
    rmse = np.linalg.norm(predictions - targets) / float(np.sqrt(n))
    return rmse 

def run_exp():
    h.init(h.v_init)
    Vvec = h.Vector()
    Vvec.record(h.cell.soma[0](0.5)._ref_v)
    h.run()
    
    return Vvec

def opt_to_epsp(params_vec):

    log_spike_time = params_vec.x[1]
    log_weight = params_vec.x[0]
    try:
        spike_time = math.exp(log_spike_time)
        weight = math.exp(log_weight)
    except:
        return 10000000
    if spike_time>SPIKE_TIME_LIM[1] or spike_time<SPIKE_TIME_LIM[0]:
        return 10000000
    if weight<0:
        return 10000000
    Stim1.start=spike_time
    for i in range(NUMBER_OF_SYNAPSES):
        Conlist[i].weight[0]=weight
    opt_v = run_exp()
    opt_v=opt_v.c(1)


    if opt_v.size() != V_DATA_Neuron.size():
        pdb.set_trace()
        print "ERROR in sizes"

    return opt_v.meansqerr(V_DATA_Neuron)






    
PATH_exp = 'ExpEPSP/Dat_Files/'
PATH_w_res = "fit_1_synapse/"
NUMBER_OF_SYNAPSES = 1
TAU_1 = 0.3
TAU_2 = 1.8
E_SYN = 0
INIT_WEIGHT = 0.01
E_PAS = -86
DELAY = 0
NUMBER_OF_SYNAPSES = 1
T_OF_INTEREST = 150
MAX_T = 400
SPIKE_TIME_LIM = [0,MAX_T]
V_DATA_Neuron = None
plot_res = 0


expname = sys.argv[1]
tree = sys.argv[2]
sec_ix = int(sys.argv[3])
seg = float(sys.argv[4])


PARALLEL_ENV = 1
h.load_file("import3d.hoc")
h.load_file("nrngui.hoc")
h("objref cell, tobj")

path ="../"
morph_file = path + "Morphs/2013_03_06_cell08_876_H41_05_Cell2.ASC"
model_file = "cell0603_08_model_cm_0_45"
model_path = path + "PassiveModels/"
print os.getcwd()
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



T_DATA,V_DATA = read_epsp_file(PATH_exp+expname+".dat")
h.dt = T_DATA[1]-T_DATA[0]
h.steps_per_ms = 1.0/h.dt
h.tstop = T_DATA[-1]

V_DATA_Neuron = h.Vector(V_DATA.size)
for i,v in enumerate(V_DATA):
    V_DATA_Neuron.x[i]=v 

Stim1 = h.NetStim()
Stim1.interval=10000 
Stim1.start=0
Stim1.noise=0
Stim1.number=1
E_PAS = np.mean(V_DATA[0:int(100/h.dt)+1])
for sec in HCell.all:
    sec.e_pas = E_PAS
h.v_init = E_PAS
INIT_Spike_time_ix  = V_DATA_Neuron.max_ind()
init_Spike_time = int(INIT_Spike_time_ix*h.dt)
print init_Spike_time

if tree == 'dend':
    sec = HCell.dend[sec_ix]
else:
    sec = HCell.apic[sec_ix]
Synlist = []
Conlist = []

Synlist.append(h.Exp2Syn(seg,sec=sec))
Synlist[0].e=E_SYN
Synlist[0].tau1=TAU_1
Synlist[0].tau2=TAU_2
Conlist.append(h.NetCon(Stim1,Synlist[0]))
Conlist[0].weight[0] = INIT_WEIGHT
Conlist[0].delay = DELAY
init_model_v = run_exp()
h.attr_praxis(0.01,1000,0)
params = h.Vector(2)


init_model_v = run_exp()
h.attr_praxis(0.01,1000,0)
params = h.Vector(2)
params.x[0] = math.log(INIT_WEIGHT)
params.x[1] = math.log(init_Spike_time)

RMSD = h.fit_praxis(opt_to_epsp,params)
RMSD = h.fit_praxis(opt_to_epsp,params)
RMSD = h.fit_praxis(opt_to_epsp,params)

print RMSD

weight = np.exp(params[0])
spike_time = np.exp(params[1])
try:
    filename_output = PATH_w_res+expname+"/"+tree+"_"+str(sec_ix)+"_"+str(seg)+".txt"
    with open(filename_output,'w') as f:
        f.write(str(NUMBER_OF_SYNAPSES)+","+str(weight)+","+str(spike_time)+","+str(RMSD))

    if plot_res:
        Conlist[0].weight[0] = weight
    
        vvec = run_exp()
        plt.plot(T_DATA,V_DATA)
        plt.plot(np.arange(0,T_DATA[-1]+0.0001,h.dt),np.array(vvec))

        plt.show()

except:
    import pdb
    pdb.set_trace()