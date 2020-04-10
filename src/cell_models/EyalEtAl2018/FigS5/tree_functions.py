########
#
# Set of functions that are useful for the analysis of neurons 
#
# AUTHOR: Guy Eyal, the Hebrew University
# CONTACT: guy.eyal@mail.huji.ac.il
#
#######


# returns the shortest path between two sections
def shortest_path(sec1,sec2,soma):

    joint_node = nearest_joint_par(sec1,sec2,soma)

    path_1 = path_to_soma(sec1,joint_node)
    path_2 = path_to_soma(sec2,joint_node)

    path = path_1+ path_2[:-1][::-1]
    return path


# returns the path from a section to a soma
def path_to_soma(sec,soma):
    path_soma = []

    temp_sec = sec
    path_soma.append(temp_sec)
    while temp_sec != soma:     
        par_sec = temp_sec.parentseg().sec
        temp_sec = par_sec
        path_soma.append(temp_sec)

    return path_soma

# returns the first branch that is on both paths of sec1 to soma and sec2 to soma
# i.e. the junction that connects sec1 and sec2
def nearest_junction(sec1,sec2,soma):
    path1_soma = path_to_soma(sec1,soma)
    path2_soma = path_to_soma(sec2,soma)


    path1_soma = path1_soma[::-1]
    path2_soma = path2_soma[::-1]

    ix = 0
    while path1_soma[ix+1] ==  path2_soma[ix+1]:
        ix+=1

    joint_node = path1_soma[ix]
    return joint_node

# returns a list with all the terminals of the model
def get_tips(cell):
    dend_tips= []
    for sec in list(cell.basal)+list(cell.apical):
        tip =1
        for child in sec.children():
            if str.find(child.hname(),'apic') >0: 
                tip =0
                break
            if str.find(child.hname(),'dend') >0 :
                tip = 0
                break
        if tip:
            dend_tips.append(sec)
    return dend_tips


# resturns a list with all the basal terminals of the model
def get_basal_tips(cell):
    dend_tips= []
    for sec in list(cell.basal):
        tip =1
        for child in sec.children():
            if str.find(child.hname(),'dend') >0 :
                tip = 0
                break
        if tip:
            dend_tips.append(sec)
    return dend_tips

