#!/usr/bin/env python

########
#
# Code to create the results for Fig 3B for cm=0.9
# In order to run it you may need to use parallel environment
# as eit runs 300*1000 simulations of around 300
# then for 1000 trials it distributes X (argv[1]) ranodm synapses on the tree
# and print to file the peak voltage and number of spikes in each trial
# 
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########

import os
MAX_SYNS = 300

CM = "09"
PARALELL_COMMAND = 'qsub'


DIR_NAME = ("DIST_"+CM+"_SYNC")
curpwd = os.getcwd()
import time
if not os.path.exists(DIR_NAME):
	os.mkdir(DIR_NAME)


current_run = time.ctime().replace(" ","_")
current_run = current_run.replace(":","_")
current_run = current_run.replace("/","_")

os.mkdir(DIR_NAME+"/"+current_run)
if not os.path.exists('Runs'):
	os.mkdir('Runs')

for i in range(MAX_SYNS,-1,-1):
    str_d = "syns_"+str(i)    
    os.makedirs(DIR_NAME+"/"+current_run+"/"+str_d)


for i in range(MAX_SYNS,-1,-1):
	outfile = 'Runs/'+current_run+'run_'+CM+'_syns_'+`i`
	txt = PARALELL_COMMAND+' -o ' + outfile + '.txt -e ' + outfile  +'err.txt' +" random_synapses_SYNC_CM"+CM+".py "+`i`+ ' '+DIR_NAME+"/"+current_run
	# txt = 'python'+" random_synapses_SYNC_cm"+CM+".py "+`i` + ' '+DIR_NAME+'/'+current_run
	print txt
	os.system(txt)
