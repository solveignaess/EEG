# Code for running hybrid simulation with EEG calculation.

To run the simulations requires hybridLFPy: 
https://github.com/INM-6/hybridLFPy/

The present example is almost identical to the example in hybridLFPy
https://github.com/INM-6/hybridLFPy/blob/master/examples/Hagen_et_al_2016_cercor/cellsim16popsParams_modified_regular_input.py
except calculation of current dipole moments have been added.

In practice, rerunning these simulations require super-computer access.

We are happy to help users getting started using the software, and for 
any problems or questions, we suggest opening an issue on the github repository, 
or contact the corresponding authors of the publications. 

The simulation is run by executing "hybrid_sim_evoked_with_EEG.py" 
with MPI. The EEG signals are calculated afterwards by executing "calculate_EEGs_from_sim.py",
and the figure is made by the script "make_figure6.py".


For full details we refer to:
Hagen et al. (2016) Hybrid scheme for modeling local field potentials from point-neuron networks. Cereb Cortex 26:4461–4496.

