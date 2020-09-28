# Biophysically detailed forward modeling of EEG and MEG signals
This is the repository for recreating all result figures from the manuscript "Biophysically detailed forward modeling of EEG and MEG signals" that is currently under review. Full details about the paper will be added when the manuscript is published.

The LaTeX code for the manuscript is in the "doc" folder, and all code files to create the figures is in the folder "src".

The software is written in Python and depends on LFPy, which in the simplest case can be installed by writing "pip install LFPy" in a terminal, see https://lfpy.rtfd.io/ for detailed instructions.



# To run the code

The code is in the "src" folder. The data needed to plot Figures 2, 3 and 7 is made and stored to
file by executing

"python3 make_data_figureX.py"

and the figures can then be made by

"python3 make_figureX.py"

Figures 4 and 5 can be made directly by

"python3 make_figureX.py"

Making Figure 6 (large-scale network simulation) is more complicated,
and in general requires super computer access.
See separate README file in the folder "hybrid_EEG_evoked" for more information.
