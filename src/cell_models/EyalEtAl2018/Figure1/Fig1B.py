########
#
# This code generates Figure 1B in Eyal 2017
# It allows to calculate the theoretical shape index for the human model cell 060308
# Then it plots both the theoretical and experimental shape indices 
# and determines the putative synapses: synapses that are in radius of 1 ms from the experimental EPSP
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
########


from neuron import h,gui
import numpy as np
import matplotlib.pyplot as plt
import pandas as pnd
import matplotlib
import progressbar


# This function creates the theoretical shape index curve and write the synaptic rise time and half width
# for each electrical segment in the model to the output_filename
def run_RTHW(WRITE_TO_FILE = 1,output_filename = "rise_and_width_synapses_locations.txt"):
	# creating the model
	h.load_file("import3d.hoc")
	h.load_file("nrngui.hoc")
	h("objref cell, tobj")
	morph_file = "../morphs/2013_03_06_cell08_876_H41_05_Cell2.ASC"
	model_file = "cell0603_08_model_cm_0_45"
	model_path = "../PassiveModels/"
	h.load_file(model_path+model_file+".hoc")
	h.execute("cell = new "+model_file+"()") 
	nl = h.Import3d_Neurolucida3()
	nl.quiet = 1
	nl.input(morph_file)
	imprt = h.Import3d_GUI(nl,0)
	imprt.instantiate(h.cell)
	HCell = h.cell
	HCell.geom_nseg()
	HCell.create_model()
	HCell.biophys()

	PLOT_MODE = 0

	TAU_1 = 0.3
	TAU_2 = 1.8
	E_SYN = 0
	WEIGHT = 0.0003 
	E_PAS = -86
	Spike_time = 10
	DELAY = 0
	NUM_OF_SYNAPSES = 1
	DT = 0.01

	h.tstop =100
	h.v_init = E_PAS
	h.steps_per_ms = 1.0/DT
	h.dt = DT

	if WRITE_TO_FILE:
		f1 = open(output_filename,'w+')

	Stim1 = h.NetStim()
	Stim1.interval=10000 
	Stim1.start=Spike_time
	Stim1.noise=0
	Stim1.number=1

	# for a given synapse, this function run the simulation  and calculate its shape index
	def calc_rise_and_width(PLOT_MODE = 0):
		Vvec = h.Vector()
		Vvec.record(HCell.soma[0](0.5)._ref_v)
		h.init(h.v_init)
		h.run()
		np_v = np.array(Vvec)
		max_idx = np.argmax(np_v)
		rise_time = max_idx*DT-Stim1.start
		half_v = E_PAS + (np_v[max_idx]-E_PAS)/2.0

		for i in range(max_idx):
			if np_v[i]>half_v:
				rise_half = i*h.dt
				break

		for i in range(max_idx,np_v.size):
			if np_v[i]<half_v:
				decay_half = i*h.dt
				break

		half_width = decay_half-rise_half

		if PLOT_MODE:
			print "rise ,",rise_time, " half width ",half_width
			np_t = np.arange(0,h.tstop+h.dt,h.dt)
			print np_v.size,np_t.size
			plt.plot(np_t,np_v,'b')
			plt.plot(np_t[max_idx],np_v[max_idx],'b*')
			plt.plot(np.array([rise_half,decay_half]),np.array([half_v,half_v]),'r')
			plt.show()

		return rise_time,half_width

	output_txt ="secanme\tx\tsoma_distance\trise_time\thalf_width\n"

	h.distance(sec=HCell.soma[0])

	# run over all the electrical segments in the model.
	# in each one of them put a synapse and run the simulation.

	print "re-creating the theoretical_RTHW"
	num_secs = len([sec for sec in HCell.all ])
	bar = progressbar.ProgressBar(max_value=num_secs, widgets=[' [', progressbar.Timer(), '] ',
                         progressbar.Bar(), ' (', progressbar.ETA(), ') ',    ])
	for ix,sec in enumerate(HCell.all):
		for seg in list(sec)+[sec(1)]:
			Syn1 = h.Exp2Syn(seg.x,sec=sec)
			Syn1.e=E_SYN
			Syn1.tau1=TAU_1
			Syn1.tau2=TAU_2
			Con1 = h.NetCon(Stim1,Syn1)
			Con1.weight[0] = WEIGHT
			Con1.delay = DELAY

			rise_time,half_width = calc_rise_and_width()
			output_txt += sec.name()+'\t'+str(seg.x)+'\t'+str(h.distance(seg.x,sec=sec))+'\t'
			output_txt += str(rise_time)+'\t'+str(half_width)+'\n'
		bar.update(ix)
			
		

	if WRITE_TO_FILE:
		f1.write(output_txt)
		f1.close()

	return output_txt.strip().split("\n")

# This function reads the shape index of the experimental EPSPs as extracted by a different code.
def read_exp_RTHW(input_filename = "ExpEPSP/expjune_14_RTHW.txt"):


	exp_RTHW = open(input_filename)
	Lines = exp_RTHW.readlines()
	exp_RTHW.close()
	exp_names = []
	exp_Taus = []
	exp_RT = []
	exp_HW = []
	for (ix,l) in enumerate(Lines[1:]):
		temp_arr=  l.strip().split('\t')
		exp_names.append(temp_arr[0])
		exp_Taus.append(float(temp_arr[1]))
		exp_RT.append(float(temp_arr[2]))
		exp_HW.append(float(temp_arr[3]))
	exp_DB = exp_names,exp_HW,exp_RT

	return exp_DB 

# This function plots fig1B according to the given experimental and theoretical shape indices
def plot_theoretical_RTHW(input_text_lines,exp_DB):
	RADIUS = 1

	CIRCLES = ['081212_1to5','110322_Sl2_Cl2_6to4','110426_Sl4_Cl2_4to6','110426_Sl4_Cl2_6to4']
	COLORS = {}
	COLORS[CIRCLES[0]]=[1,0,0.22]
	COLORS[CIRCLES[1]]=[1, 0.6 , 0.78]
	COLORS[CIRCLES[2]]=[0, 0.5, 1]
	COLORS[CIRCLES[3]]=[1 ,204.0/255.0, 0]

	exp_names = exp_DB[0]
	exp_HW = exp_DB[1]
	exp_RT = exp_DB[2]

	sec_names = []
	segs = []
	dist = []
	RT = []
	HW = []
	first_apic = 0
	for (ix,l) in enumerate(input_text_lines[1:]):
		temp_arr=  l.strip().split('\t')
		sec_names.append(temp_arr[0])
		if str.find(temp_arr[0],'apic')>-1 and not first_apic:
			first_apic= ix

		segs.append(float(temp_arr[1]))
		dist.append(float(temp_arr[2]))
		RT.append(float(temp_arr[3]))
		HW.append(float(temp_arr[4]))

	np_RT = np.array(RT)
	np_HW = np.array(HW)

	theoretical_RTHW = [sec_names,segs,np_RT,np_HW]

	fig, ax = plt.subplots()
	patch_circles = []
	for ix,name in enumerate(exp_names):
		if name in CIRCLES:
			patch_circles.append(matplotlib.patches.Circle((exp_RT[ix],exp_HW[ix]),radius=RADIUS,color=COLORS[name],alpha=0.5))

	for patch in patch_circles:
		fig.axes[0].add_patch(patch)

	plt.scatter(np_RT[first_apic:],np_HW[first_apic:],s=30,c=dist[first_apic:],marker='o',cmap='hsv',
								linewidths=0,norm=matplotlib.colors.Normalize(0,max(dist)))
	plt.scatter(np_RT[:first_apic],np_HW[:first_apic],s=30,c=dist[:first_apic],marker='D',cmap='hsv',
								edgecolors='k',norm=matplotlib.colors.Normalize(0,max(dist)))




	plt.xlabel('Rise time (ms)',fontsize=18)
	plt.ylabel('Half width (ms)',fontsize=18)
	plt.tick_params(labelsize=16)
	cbar=plt.colorbar()
	cbar.set_label('Distance from soma (um)', fontsize=18)
	cbar.ax.tick_params(labelsize=16)

	plt.scatter(np.array(exp_RT),np.array(exp_HW),s=30,c='k')


	fig, ax = plt.subplots()
	patch_circles = []
	for ix,name in enumerate(exp_names):
		if name in CIRCLES:
			patch_circles.append(matplotlib.patches.Circle((exp_RT[ix],exp_HW[ix]),radius=RADIUS,color=COLORS[name],alpha=0.5))

	for patch in patch_circles:
		fig.axes[0].add_patch(patch)

	plt.scatter(np_RT[first_apic:],np_HW[first_apic:],s=30,c=dist[first_apic:],marker='o',cmap='hsv',
								linewidths=0,norm=matplotlib.colors.Normalize(0,max(dist)))
	plt.scatter(np_RT[:first_apic],np_HW[:first_apic],s=30,c=dist[:first_apic],marker='D',cmap='hsv',
								edgecolors='k',norm=matplotlib.colors.Normalize(0,max(dist)))

	plt.xlabel('Rise time (ms)',fontsize=18)
	plt.ylabel('Half width (ms)',fontsize=18)
	plt.tick_params(labelsize=16)
	cbar=plt.colorbar()
	cbar.set_label('Distance from soma (um)', fontsize=18)
	cbar.ax.tick_params(labelsize=16)

	plt.scatter(np.array(exp_RT),np.array(exp_HW),s=30,c='k')
	ax.axis([1,9,12,20])

	plt.show()
	return theoretical_RTHW

def in_circle(RT1,HW1,RT2,HW2,radius =1):
	return ((RT1-RT2)**2+(HW1-HW2)**2)<radius**2

# This function finds theoretical putative synapses for each experimental EPSP
def putative_synapses(exp_DB,theoretical_RTHW,WRITE_TO_FILE=1,output_filename='putative_synapses.txt'):
	RADIUS = 1

	exp_names = exp_DB[0]
	exp_HW = exp_DB[1]
	exp_RT = exp_DB[2]
	
	model_sec_names = theoretical_RTHW[0]
	model_segs = theoretical_RTHW[1]
	model_RT = theoretical_RTHW[2]
	model_HW = theoretical_RTHW[3]

	row_list = []
	for exp_ix,expname in enumerate(exp_names):

		for model_ix,secname in enumerate(model_sec_names):
			if in_circle(exp_RT[exp_ix],exp_HW[exp_ix],model_RT[model_ix],model_HW[model_ix],RADIUS):
				sec = secname.split('.')[1]
				sec_split = sec.split('[')
				sec_tree = sec_split[0]
				sec_ix = int(sec_split[1][:-1])

				dict1 = {'exp_name':expname,'sec_tree':sec_tree,'sec_ix':sec_ix,'seg':model_segs[model_ix]}
				row_list.append(dict1)
	df = pnd.DataFrame(row_list,columns = ['exp_name','sec_tree','sec_ix','seg'])  

	if WRITE_TO_FILE:
		s = ""
		for exp_ix,expname in enumerate(exp_names):
			s += expname +"\n"
			df_exp = df[df['exp_name']==expname]
			s += "trees = "+str(list(df_exp.sec_tree))+"\n"
			s += "secs = "+str(list(df_exp.sec_ix))+"\n"
			s += "segs = "+str(list(df_exp.seg))+"\n"

		with open(output_filename,'w+') as f:
			f.write(s)






def plot_Fig1B(re_create_model_RTHW = False,RTHW_file = "rise_and_width_synapses_locations.txt",re_create_putative_syns = False):
	'''

	plotting fig1B
	re_create_model_RTHW -- re runs the simulation that activates a synapse in each electrical segment of the cell
		if False, reads the shape index from RTHW_file
	RTHW_file -- the file to read from/ write to the theoretical shape index
	re_create_putative_syns -- re-creating the putative synapses file
	''' 


	if re_create_model_RTHW:
		input_text_lines = run_RTHW()
	else:
		model_RTHW = open(RTHW_file)
		input_text_lines = model_RTHW.readlines()
		model_RTHW.close()


	exp_DB = read_exp_RTHW()

	theoretical_RTHW = plot_theoretical_RTHW(input_text_lines,exp_DB)

	if re_create_putative_syns:
		putative_synapses(exp_DB,theoretical_RTHW)

plot_Fig1B()






	




