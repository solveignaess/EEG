#!/usr/bin/env python

######
# Run all the scripts in this folder in parallel
# chnage the PARALELL_COMMAND to fit your cluster
######


import sys,os
os.system('nrnivmodl ../../mechanisms/')

PARALELL_COMMAND = 'qsub'
outfile = 'Runs_nmda_spike'
errfile = 'Runs_nmda_spike_err'


for model in ["0603_03","0603_08","0603_11","1303_03","1303_05","1303_06"]:
    s = " ".join([PARALELL_COMMAND,'-o',outfile+model+".txt",'-e',errfile+model+".txt"])
    s1 = s + " num_syn_per_segment_nmda_spike_"+model+".py"
    print s1
    os.system(s1)

    s2 = s + " num_syn_per_segment_nmda_spike_"+model+"_terminals.py"
    print s2
    os.system(s2)