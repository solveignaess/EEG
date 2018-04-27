import matplotlib
matplotlib.rc('pdf', fonttype=42)
import pylab as plt
from matplotlib.colors import LogNorm

plt.rcParams.update({
    'xtick.labelsize': 11,
    'xtick.major.size': 5,
    'ytick.labelsize': 11,
    'ytick.major.size': 5,
    'font.size': 15,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'legend.fontsize': 14,
    'figure.subplot.wspace': 0.4,
    'figure.subplot.hspace': 0.4,
    'figure.subplot.left': 0.1,
})

elec_color = '#00d2ff'
res_color = '#0080ff'
reg_color = '#ff0000'
pas_color = 'k'
syn_color = '#00cc00'
cell_color = '#c4c4c4'



apic_qa_clr_dict = {-0.5: "pink",
               0.0: pas_color,
               2.0: "lightblue"}

qa_clr_dict = {-0.5: reg_color,
               None: "k",
               0.0: pas_color,
               2.0: res_color}

cond_clr = {'active': 'r',
               'passive': 'k',
               'Ih': 'b',
                'Ih_frozen': 'c',
            'Ih_plateau': 'orange',
            'Ih_plateau2': 'red',
            None: "k",
            -0.5: reg_color,
            0.0: pas_color,
            2.0: res_color,
            }

cond_clr_mm = {'active': 'r',
               'passive': 'k',
               'Ih': 'b',
                'Ih_frozen': 'c',


               None: "0.5",
               -0.5: reg_color,
               0.0: pas_color,
               2.0: 'lightblue',
               }

cond_names = {-0.5: 'passive+regenerative',
              0.0: 'passive+frozen',
              2.0: 'passive+restorative',
              None: "passive",
                     'active': 'active',
                     'passive': 'passive',
                     'Ih': 'passive + I$_h$',
                     'Ih_frozen': 'passive + frozen I$_h$',
                    'Ih_plateau': 'passive + I$_h$ early plateau',
                    'Ih_plateau2': 'passive + I$_h$ late plateau',
              }
def mark_subplots(axes, letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ', xpos=-0.12, ypos=1.15):

    if not type(axes) is list:
        axes = [axes]

    for idx, ax in enumerate(axes):
        ax.text(xpos, ypos, letters[idx].capitalize(),
                horizontalalignment='center',
                verticalalignment='center',
                fontweight='demibold',
                fontsize=12,
                transform=ax.transAxes)

def simplify_axes(axes):

    if not type(axes) is list:
        axes = [axes]

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

def color_axes(axes, clr):
    if not type(axes) is list:
        axes = [axes]
    for ax in axes:
        ax.tick_params(axis='x', colors=clr)
        ax.tick_params(axis='y', colors=clr)
        for spine in list(ax.spines.values()):
            spine.set_edgecolor(clr)

