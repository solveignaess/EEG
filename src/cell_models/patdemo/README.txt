This readme file gives usage instructions for patdemo2.zip
(essentially patdemo.zip at http://www.cnl.salk.edu/~zach/methods.html
by Z. F. Mainen and T. J. Sejnowski) (see bottom for important updates
this file includes)

This model contains compartmental models of four reconstructed
neocortical neurons (Layer 3 Aspiny, layer 4 Stellate, layer 3 and
layer 5 Pyramidal neurons) with active dendritic currents using
NEURON. It is shown here that an entire spectrum of firing patterns
can be reproduced in this set of model neurons which share a common
distribution of ion channels and differ only in their dendritic
geometry.

The reference paper is:
Z. F. Mainen and T. J. Sejnowski (1996) Influence of dendritic
structure on firing pattern in model neocortical neurons.
Nature 382:363-366.

See also http://www.cnl.salk.edu/~zach/methods.html and
http://www.cnl.salk.edu/~zach/

This package is written in the NEURON simulation program written by
Michael Hines and available on internet at:
http://www.neuron.yale.edu/


  HOW TO RUN (under NEURON version 4 and higher)
  =========================================

To compile the demo, NEURON and INTERVIEWS must be installed and
working on the machine you are using.

When the patdemo1.zip file is unzipped it creates a patdemo directory
which contains the hoc and mod NEURON program files.  Change directory
to patdemo.

under UNIX:
===========

Just type "nrnivmodl" to compile the mechanisms given in the mod files
in the patdemo directory.

Execute the first figure demo program by typing:

nrngui demofig1.hoc

continue below under back to any platform:

under MS WINDOWS (PC):
======================

Press Start button (lower left corner) and then press Programs and
then NEURON and then mknrndll DOS box.  Change directory to where the
zip file was unzipped and enter the directory that came with the zip
file (patdemo). Type mknrndll and press the Enter key.

Execute the first figure demo program by typing:

nrngui

In the NEURON main menu click on file and open and then double click
on demofig1.hoc

continue below:

back to any platform:
=====================

For each of the cells, click on one of the cell buttons in the
figure 1 window - for example

a. L3 Aspiny

and then click on the Init & Run button to observe the voltage trace.

to explore the figure 2 model quit (click on file and then quit in the
NEURON main menu) and start up again using the command:

nrngui demofig2.hoc

To explore parameters in these models, on the NEURON main menu click
on Tools, Distributed Mechanisms, Viewers, and Name Values

When the special window comes up double click on one of the sections
in the left hand column, e.g. soma and then finally you can
observe/change the parameters of the model.  Note for example that
changing the morphology of the model, e.g. changing L, will cause the
model to change shape in the Shape window.  If you closed the shape
window press one of the cell buttons again in the Figure 1 window.  At
any time you can test how any of your modifications change the
electrical excitability by pressing Init & Run.

If you wish to change the current injected into the model, on the
NEURON main menu click on Tools, Point Processes, Viewers, and IClamp.
Then on the new IClamp window double click on "soma(0.5)".  The on the
"IClamp[] at soma(.5)" modify the protocol by for example changing the
amplitude (amp) of the current to a value that is for example .02 nA
larger.  To compare with the previous run of the model right click on
the graph of v(.5), and select the menu item "Keep Lines".  Notice
that there are many ways to to zoom/scroll the graph by right clicking
on the graph and slide to the top of the pop-up menu to the
View... menu and then slide off to the right and select one of these
items.

---
Version 2002-08-30 contains modifications for variable time step method.
Version 2011-02-02 makes all mod files threadsafe.  For cad.mod this 
required replacing euler by cnexp, which changes spike times slightly
but has no effect on qualitative results.
Version 2012-01-05 derivimplicit was used in place of cnexp for kca and
cad mechanisms (better for mechanisms with ion's)
Version 2012-01-06 cnexp restored for kca.mod because linear in the
state variables.
Version 2012-05-15 updates from Ted Carnevale to make THREADSAFE,
clean code, and handle singularities correctly (ca.mod, km.mod, and
kv.mod didn't).
