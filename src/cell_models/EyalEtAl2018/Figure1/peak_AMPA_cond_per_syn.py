########
#
# This code summarizes the fit results into a table, similar to table S1 in Eyal et al 2017.
#
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########

import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pnd



# Read all the fit files and create one file from all of them
def group_fits_1_syn(PATH_w_1_syn,putative_syn_file = 'putative_synapses.txt',TAU_1 = 0.3,TAU_2 = 1.8):
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

        fit_files = glob.glob(PATH_w_1_syn+expname+"/*.txt")
        s = ""
        for f_name in fit_files:
            syn_pars = f_name.split("/")[-1].split(".txt")[0]
            synapse = syn_pars.split("_")
            synapse[2] =  '%.3f' % float(synapse[2])
            with open(f_name) as f_syn:
            
                syn_lines = f_syn.readlines()
            opt_res = syn_lines[0].split(",")
            opt_res[1] = '%.6f' % float(opt_res[1])
            opt_res[2] = '%3.3f' % float(opt_res[2])
            opt_res[3] = '%.6f' % float(opt_res[3])
            output_txt = [str(s1) for s1 in [opt_res[0],expname,synapse[0],synapse[1],synapse[2],TAU_1,TAU_2,opt_res[1],opt_res[2],opt_res[3]]]
            s += "\t".join(output_txt)+"\n"
        
        with open(PATH_w_1_syn+"fit_1_syn_"+expname+".txt",'w+') as f_w:
            f_w.write("NUMBER_OF_SYNAPSES\tfit_to_EPSP\tSEC_List\tSec_Index\tSec_Seg\tTAU_1\tTAU_2\tWEIGHT\tSpike_time\tRMSD\n")
            f_w.write(s)



# Read all the fit files and create one file from all of them
def group_fits_5_syn(PATH_w_5_syn,putative_syn_file = 'putative_synapses.txt',TAU_1 = 0.3,TAU_2 = 1.8):
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

        fit_files = glob.glob(PATH_w_5_syn+expname+"/*.txt")
        s = ""
        for f_name in fit_files:
            with open(f_name) as f_syn:
            
                syn_lines = f_syn.readlines()
            opt_res = syn_lines[0].split(",")
            num_of_syns = int(opt_res[0])
            syns = []
            for j in range(num_of_syns):
                syns.append(opt_res[1+j*3]) # tree
                syns.append(opt_res[1+j*3+1]) # section
                syns.append(opt_res[1+j*3+2]) # segment

            weight = '%.6f' % float(opt_res[-3]) # weight
            spike_time = '%3.3f' % float(opt_res[-2])
            RMSD = '%.6f' % float(opt_res[-1])
            output_txt = [str(s1) for s1 in [num_of_syns,expname]+syns + [TAU_1,TAU_2,weight,spike_time,RMSD]]
            s += "\t".join(output_txt)+"\n"
        
        with open(PATH_w_5_syn+"fit_5_syn_"+expname+".txt",'w+') as f_w:

            f_w.write("NUMBER_OF_SYNAPSES\tfit_to_EPSP\t")
            for i in range(num_of_syns):
                f_w.write("SEC_List\tSec_Index\tSec_Seg\t")
            f_w.write("TAU_1\tTAU_2\tWEIGHT\tSpike_time\tRMSD\n")
            f_w.write(s)




# Read fit summaries and calculate the average peak AMPA conductance per 
# experimental synapse
def create_table(folder,file_prefix):
    fit_files = glob.glob(folder+file_prefix+"*.txt")
    A = []
    for f in fit_files:
        try:
            table = pnd.read_csv(f,delimiter = "\t")
            if len(table['WEIGHT'])== 0:
                continue

            expname = f[len(folder)+len(file_prefix)+1:-4]
            print "%s\t%f\t%f"%(expname,np.mean(table['WEIGHT']*1000),np.std(table['WEIGHT']*1000))
            A.append(np.mean(table['WEIGHT']*1000))
        except:
            continue

    print "total"
    print np.mean(A),"+-",np.std(A)
    return np.mean(A),np.std(A)



PATH_w_1_syn = "fit_1_synapse/"
PATH_w_5_syn = "fit_5_synapses/"

if __name__ == "__main__":
    group_fits_1_syn(PATH_w_1_syn)
    group_fits_5_syn(PATH_w_5_syn)

    fit_1_syn = create_table(PATH_w_1_syn,"fit_1_syn")
    fit_5_syn = create_table(PATH_w_5_syn,"fit_5_syn")






