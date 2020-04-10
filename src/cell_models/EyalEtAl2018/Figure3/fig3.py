########
#
# This code generates Figure 3 in Eyal 2017
# It reads the outputs of simulating_spines_on_cell0603_08.py 
# which simulates an activation of a one spine in each electrical compartment (separately) of human model 060308.
# The code here present the peak voltage (for each simulation) in the spine head, spine base and soma. 
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########



import numpy as np
import matplotlib.pyplot as plt
import pdb
import pandas as pnd
import glob

c_spine = [181.0/256,170.0/256,31.0/256]
c_shaft = [0.0,0.5,0.0]
c_soma = [0.0, 174.0/256,239.0/256]

E_PAS = -86

# analyze the results simulating human spines for the six human models (simulating_spines_on_all_models.py)
def analyze_all_models(path = "human_spines_simulations/"):
    spine_res_files = glob.glob(path+"*.txt")
    soma_v_w_spine = []
    shaft_v_w_spine = []
    spine_v_w_spine = []
    for f in spine_res_files:
        df = pnd.read_csv(f,delimiter=',')
        df_w_spine = df[df['Spine']==1]

        df_no_spine = df[df['Spine']==0]

        soma_v_w_spine += list(df_w_spine['max_soma_v']-E_PAS)
        shaft_v_w_spine += list(df_w_spine['max_shaft_v']-E_PAS)
        spine_v_w_spine += list(df_w_spine['max_spine_v']-E_PAS)

    spine_v_w_spine = np.array(spine_v_w_spine)
    shaft_v_w_spine = np.array(shaft_v_w_spine)
    soma_v_w_spine = np.array(soma_v_w_spine)

    print 'average spine voltage in the six models',str(np.mean(spine_v_w_spine)),' +- ',str(np.std(spine_v_w_spine))
    print 'average spine voltage in the six models',str(np.mean(shaft_v_w_spine)),' +- ',str(np.std(shaft_v_w_spine))
    print 'average soma voltage in the six models',str(np.mean(soma_v_w_spine)),' +- ',str(np.std(soma_v_w_spine))

    print 'average attenutation from spine head to spine base:',str(np.mean(spine_v_w_spine/shaft_v_w_spine)),' +- ',str(np.std(spine_v_w_spine/shaft_v_w_spine))

    print 'average attenutation from spine head to soma:',str(np.mean(spine_v_w_spine/soma_v_w_spine)),' +- ',str(np.std(spine_v_w_spine/soma_v_w_spine))



def plot_fig_2():
   
    # to re run the simulations reported in synapses_on_spines_cell0603_08.txt, run simulating_spines_on_cell0603_08.py
    df = pnd.read_csv('synapses_on_spines_cell0603_08.txt',delimiter=',')
    df_w_spine = df[df['Spine']==1]
    df_no_spine = df[df['Spine']==0]



    soma_v_w_spine = df_w_spine['max_soma_v']-E_PAS
    shaft_v_w_spine = df_w_spine['max_shaft_v']-E_PAS
    spine_v_w_spine = df_w_spine['max_spine_v']-E_PAS
    IR_shaft = df_w_spine['ir_shaft']

    EXAMPLE_CASE = 1112 # the case of figure 3B

    fig = plt.figure()
    plt.scatter(IR_shaft,spine_v_w_spine,c=c_spine,s=5,marker = 'o')
    plt.scatter(IR_shaft,shaft_v_w_spine,c=c_shaft,s=5,marker = 'o')
    plt.scatter(IR_shaft,soma_v_w_spine,c=c_soma,s=5,marker = 'o')
    plt.scatter(IR_shaft[EXAMPLE_CASE],spine_v_w_spine[EXAMPLE_CASE],c='k',s=15,marker = 'o')
    plt.scatter(IR_shaft[EXAMPLE_CASE],shaft_v_w_spine[EXAMPLE_CASE],c='k',s=15,marker = 'o')
    plt.scatter(IR_shaft[EXAMPLE_CASE],soma_v_w_spine[EXAMPLE_CASE],c='k',s=15,marker = 'o')


    plt.tick_params(direction = 'in')
    plt.xlim(0,1100)
    plt.ylim(0,23)


    fig = plt.figure()
    plt.scatter(IR_shaft,soma_v_w_spine,c=c_soma,s=5,marker = 'o')
    plt.scatter(IR_shaft[EXAMPLE_CASE],soma_v_w_spine[EXAMPLE_CASE],c='k',s=15,marker = 'o')


    # Read the simulation results of the example case in Fig3B. To re run it, run fig3B.py
    T = pnd.read_csv('example_stimulation of_a_spine_connected_to_dend_62.txt',
                    skiprows=1,delimiter = ' ',header=None)
    tvec = T[0]
    v_soma = T[1]
    v_shaft = T[2]
    v_spine = T[3]
    fig = plt.figure()
    plt.plot(tvec,v_spine,c=c_spine)
    plt.plot(tvec,v_shaft,c=c_shaft)
    plt.plot(tvec,v_soma,c=c_soma)
    plt.xlim(0,50)

    analyze_all_models()
        
plot_fig_2()




