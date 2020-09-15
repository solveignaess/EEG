# NY_head
EEG simulations with New York head model

Calculate EEG signals based on the head model from: https://www.parralab.org/nyhead/

Requires the file https://www.parralab.org/nyhead/sa_nyhead.mat

Depends on LFPy for comparison with four-sphere head model.

Includes a conda environment file, which was needed to plot the head model in mayavi.

To execute the simulations and store results to file, run 
"python3 make_data_figure6.py"
and to make the figure,
"python3 make_figure6.py"

