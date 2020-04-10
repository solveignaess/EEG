README

Finding putative synapses for the human model 0603 cell08 and fitting the synaptic conductances that fit the putative synapses
the experimental data is in ExpEPSP
The scripts Fig1B.py, Fig1C.py and Fig1D.py re-creates the figures. 
parallel_fit_exp_synapses.py re runs the fits (see comments in the file, may require parallel environment) 
putative_synapses.txt is the summary of the putative synapses results (when putative synapse is defined as having a shape index that is within 1 ms radius from the experimental shape index)
rise_and_width_synapses_locations.txt has the theoretical shape indices as results from running Fig1B (with re_create_model_RTHW = True)
