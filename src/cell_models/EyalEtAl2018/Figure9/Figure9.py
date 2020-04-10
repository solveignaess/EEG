########
#
# This code generates Figure 9 in Eyal et al 2017.
# The calculations here are based on the theory by Poirazi and Mel 2001 and the values 
# from human and rodent neurons found in Eyal et al 2017
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########


import numpy as np
from scipy import special
from math import *
import matplotlib.pyplot as plt

range_subunits = range(1,31,1)

human_subunits = 25
rat_subunits = 14


range_synapses = range(0,40000)
human_synapses = 30000
rat_synapses = 10000
contacts_per_syn = 5 #Average value

color_rat = np.array([139,69,19])/256.0
color_human = [1,0,0]

# As in Poirazi and Mel 2001 eq4
def capacity_1lm(synapses,affernts):
    ln_val = special.gammaln(synapses+affernts) - special.gammaln(synapses+1) - special.gammaln(affernts)
    return 2*ln_val/log(2.0)

# As in Poirazi and Mel 2001 eq5
def capacity_2lm(m_subunits,k_syn_in_subunits,d_input_lines):

    inner_comb = special.comb(k_syn_in_subunits+d_input_lines-1,k_syn_in_subunits,exact=True)
    try:
        ln_val = special.gammaln(inner_comb+m_subunits) - special.gammaln(m_subunits+1) - special.gammaln(inner_comb)
    except:
        diff_a = sum([log(i) for i in range(inner_comb,inner_comb+m_subunits)]) # since lngamma(x) = lngamma(x+1)-ln(x)
        ln_val = diff_a-special.gammaln(m_subunits+1)
    return 2*ln_val/log(2.0)




def plot_1lm_fig(contacts_per_syn,human_synapses,rat_synapses):
    plt.figure()
    capacity_arr = [capacity_1lm(synapses,synapses/contacts_per_syn) for synapses in range_synapses]
    plt.scatter([rat_synapses],[capacity_1lm(rat_synapses,rat_synapses/contacts_per_syn)],c=color_rat)
    plt.scatter([human_synapses],[capacity_1lm(human_synapses,human_synapses/contacts_per_syn)],c=color_human)
    plt.plot(range_synapses,capacity_arr,c='k')
    plt.xlabel('number of synapses')
    plt.ylabel('capacitiy')
    plt.title('1LM with '+str(contacts_per_syn)+" contacts per synapse")



def plot_2lm_fig(contacts_per_syn,human_synapses,rat_synapses,human_subunits,rat_subunits):

    capacity_rat_synapses = []
    for num_of_subunits in range_subunits:
        afferents_rat = rat_synapses/contacts_per_syn
        k_syn_in_subunits = rat_synapses/num_of_subunits
        capacity_rat_synapses.append(capacity_2lm(num_of_subunits,k_syn_in_subunits,afferents_rat))
    
    plt.plot(range_subunits,capacity_rat_synapses,c='k')

    capacity_human_synapses = []
    for num_of_subunits in range_subunits:
        afferents_human = human_synapses/contacts_per_syn
        k_syn_in_subunits = human_synapses/num_of_subunits
        capacity_human_synapses.append(capacity_2lm(num_of_subunits,k_syn_in_subunits,afferents_human))
    
    plt.plot(range_subunits,capacity_human_synapses,c='k')

    plt.scatter([rat_subunits],[capacity_2lm(rat_subunits,rat_synapses/rat_subunits,afferents_rat)],c=color_rat)
    plt.scatter([human_subunits],[capacity_2lm(human_subunits,human_synapses/human_subunits,afferents_human)],c=color_human)

    plt.xlabel('number of subunits')
    plt.ylabel('capacitiy')



plot_1lm_fig(contacts_per_syn,human_synapses,rat_synapses)
plt.xlim(-1000,42000)
plt.ylim(-2000,65000)
plt.figure()
plot_2lm_fig(contacts_per_syn,human_synapses,rat_synapses,human_subunits,rat_subunits)
plt.xlim(-1,32)
plt.ylim(-9000,270000)



