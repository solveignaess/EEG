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
pas_color = 'k'
syn_color = '#00cc00'
cell_color = '#c4c4c4'


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

