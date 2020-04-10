########
#
# This code allows to recreate the fit values for Figure 1 in Eyal et al.
# It runs over all the putative synapses as defined and created in Fig1B 
# and for each one of them fit both the case of one contact per synapses and the case of five contacts per synapses
# This is the script that runs the many fits. I use here a parallel environment. It is possible also to run it 
# on one computer but it may take up to few days.
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########



import numpy as np
import matplotlib.pyplot as plt
import pandas as pnd
import matplotlib
import os
import sys
import subprocess
import glob

PATH_w_1_syn = "fit_1_synapse/"
PATH_w_5_syn = "fit_5_synapses/"

PARALLEL_ENV = 0



def run_cmd1(expname,tree,sec,seg):
    if PARALLEL_ENV:
        PARALELL_COMMAND = 'qsub'
        s1 = expname+"_"+tree+"_"+str(sec)+"_"+str(seg)
        outfile = 'Runs/'+str(s1)+'.txt'
        errfile = 'Runs/'+str(s1)+'err.txt'
        s = " ".join([PARALELL_COMMAND,'-o',outfile,'-e',errfile])

    else:
        s = 'python '

    s += ' fit_synapse.py '+expname+" " +tree+ " "+ str(sec)+ " "+str(seg)
    return s

def run_cmd5(expname,c,list_of_segments):
    if PARALLEL_ENV:
        PARALELL_COMMAND = 'qsub'
        s1 = expname+"_"+str(c)
        outfile = 'Runs/'+str(s1)+'.txt'
        errfile = 'Runs/'+str(s1)+'err.txt'
        s = " ".join([PARALELL_COMMAND,'-o',outfile,'-e',errfile])

    else:
        s = 'python '

    list_of_str_segs = []
    for seg in list_of_segments:

        list_of_str_segs += [" ".join([seg[0],str(seg[1]),str(seg[2])])]

    s += ' fit_5_synapses.py '+expname+" " +str(c)+" "+" ".join(list_of_str_segs)
    return s



def run_fit_1_synapse(write_to_files=1,putative_syn_file = 'putative_synapses.txt'):


    f = open(putative_syn_file)
    lines = f.readlines()
    f.close()
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
            secs_str = secs_str.split("[")[1]
            secs = secs_str[:-1].split(",")
            secs = [int(s) for s in secs]
            segs_str = segs_str.split("[")[1]
            segs = segs_str[:-1].split(",")
            segs = [float(s) for s in segs]
        except:
            import pdb
            pdb.set_trace()
        if not os.path.exists(PATH_w_1_syn+expname):
            os.mkdir(PATH_w_1_syn+expname)
        for jx in range(len(trees)):
            s = run_cmd1(expname,trees[jx][1:-1],secs[jx],segs[jx])
            print s
            os.system(s)
            
        



def run_fit_5_synapse(write_to_files=1,putative_syn_file = 'putative_synapses.txt',NUM_OF_ITERATIONS = 150):


    f = open(putative_syn_file)
    lines = f.readlines()
    f.close()
    ix = 0

    NUM_OF_CONTACTS = 5

    while ix<len(lines):
        expname = lines[ix].strip()
        trees_str = lines[ix+1].strip()
        secs_str = lines[ix+2].strip()
        segs_str = lines[ix+3].strip()
        ix+=4

        try:
            trees_str = trees_str.split("[")[1]
            trees = trees_str[:-1].split(", ")
            secs_str = secs_str.split("[")[1]
            secs = secs_str[:-1].split(",")
            secs = [int(s) for s in secs]
            segs_str = segs_str.split("[")[1]
            segs = segs_str[:-1].split(",")
            segs = [float(s) for s in segs]
        except:
            import pdb
            pdb.set-trace()
        if not os.path.exists(PATH_w_5_syn+expname):
            os.mkdir(PATH_w_5_syn+expname)
        for jx in range(NUM_OF_ITERATIONS):
            rand_syns = [np.random.randint(len(trees)) for i in range(NUM_OF_CONTACTS)]
            list_of_synapses = []
            for rs in rand_syns:
                list_of_synapses.append([trees[rs][1:-1],secs[rs],segs[rs]])
            s = run_cmd5(expname,jx,list_of_synapses)
            print s

            os.system(s)
            
        
    


run_fit_1_synapse()

run_fit_5_synapse()
 






