########
#
# This code generates Figure 1C in Eyal 2017
# The code reads the location of the putative synapses according to figure 1B
# and present it on the morphology of Human cell 060308
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


TAU_1 = 0.3
TAU_2 = 1.8
E_SYN = 0

ThickFactor = 2


COLOR_BLUE = 3 
COLOR_ORANGE = 5 
COLOR_MAGENTA = 7
COLOR_RED = 2

STYLE = 4 
SIZE = 8

# read the putative synapses file
def putative_synapses(putative_syn_file = 'putative_synapses.txt'):
    f = open(putative_syn_file)
    lines = f.readlines()
    f.close()
    putative_dict = {}
    ix = 0
    while ix<len(lines):
        expname = lines[ix].strip()
        trees_str = lines[ix+1].strip()
        secs_str = lines[ix+2].strip()
        segs_str = lines[ix+3].strip()
        ix+=4
        print expname
        try:
            trees_str = trees_str.split("[")[1]
            trees = trees_str[:-1].split(", ")
            trees = [t[1:-1] for t in trees]
            secs_str = secs_str.split("[")[1]
            secs = secs_str[:-1].split(",")
            secs = [int(s) for s in secs]
            segs_str = segs_str.split("[")[1]
            segs = segs_str[:-1].split(",")
            segs = [float(s) for s in segs]
            putative_dict[expname] = [trees,secs,segs]
        except:
            continue
    return putative_dict

# plot the synapses on the NEURON morphology
def plot_synapses(shp,putative_dict,expname,Synlist,c):
    putative_syns = putative_dict[expname]
    trees = putative_syns[0]
    secs = putative_syns[1]
    segs = putative_syns[2]

    for i in range(len(segs)):

        if trees[i] == 'dend':
            Synlist.append(h.Exp2Syn(segs[i],sec=HCell.dend[secs[i]]))
        else:
            Synlist.append(h.Exp2Syn(segs[i],sec=HCell.apic[secs[i]]))
        Synlist[-1].e=E_SYN
        Synlist[-1].tau1=TAU_1
        Synlist[-1].tau2=TAU_2
        shp.point_mark(Synlist[-1],c,STYLE,SIZE)

def plot_fig1C1(putative_dict,Synlist):

    plot_synapses(shp,putative_dict,"081212_1to5",Synlist,COLOR_RED)
    plot_synapses(shp,putative_dict,"110426_Sl4_Cl2_4to6",Synlist,COLOR_MAGENTA)


def plot_fig1C2(putative_dict,Synlist):

    plot_synapses(shp2,putative_dict,"110426_Sl4_Cl2_6to4",Synlist,COLOR_BLUE)
    plot_synapses(shp2,putative_dict,"110322_Sl2_Cl2_6to4",Synlist,COLOR_ORANGE)



Synlist = []
for sec in HCell.basal:
    sec.diam = sec.diam*ThickFactor
for sec in HCell.apical:
    sec.diam = sec.diam*ThickFactor

putative_dict = putative_synapses()
shp = h.Shape()
shp.show(0)
shp.rotate(0,0,0,0,0,2.8)
shp.view(-400,-600,1000,1700,0,0,800,800)
plot_synapses(shp,putative_dict,"081212_1to5",Synlist,COLOR_RED)
plot_synapses(shp,putative_dict,"110426_Sl4_Cl2_4to6",Synlist,COLOR_MAGENTA)

shp2 = h.Shape()
shp2.show(0)
shp2.rotate(0,0,0,0,0,2.8)
shp2.view(-400,-600,1000,1700,400,0,800,800)
plot_synapses(shp2,putative_dict,"110426_Sl4_Cl2_6to4",Synlist,COLOR_BLUE)
plot_synapses(shp2,putative_dict,"110322_Sl2_Cl2_6to4",Synlist,COLOR_ORANGE)
