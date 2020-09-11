"""
conda activate oldqt
python3 plot_head_model_mayavi.py
This script was used for making illustrations of head and brain in the folder
"head_model_illustrations"
"""

import numpy as np
from traits.api import HasTraits, Instance
from traitsui.api import View, Item, HSplit, Group
from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor


class MyDialog(HasTraits):

    scene1 = Instance(MlabSceneModel, ())
    scene2 = Instance(MlabSceneModel, ())

    def draw_scene1(self):
        mlab.clf(figure=self.scene1.mayavi_scene)
        mlab.triangular_mesh(self.ny_head.head[0, :], self.ny_head.head[1, :], self.ny_head.head[2, :],
                             self.ny_head.head_tri.T,
                             color=self.head_color, figure=self.scene1.mayavi_scene)
        nodes = mlab.points3d(self.ny_head.elecs[0, :],
                              self.ny_head.elecs[1, :],
                              self.ny_head.elecs[2, :],
                              vmax=self.vmax, vmin=self.vmin, colormap="bwr",
                              scale_factor=10.0, figure=self.scene1.mayavi_scene)
        nodes.glyph.scale_mode = 'scale_by_vector'
        nodes.mlab_source.dataset.point_data.scalars = self.ny_head.eeg
        # mlab.colorbar(nodes)

    def draw_scene2(self):
        mlab.clf(figure=self.scene2.mayavi_scene)
        mlab.triangular_mesh(self.ny_head.cortex[0, :], self.ny_head.cortex[1, :],
                             self.ny_head.cortex[2, :], self.ny_head.cortex_tri.T,
                             color=self.head_color, figure=self.scene2.mayavi_scene)

        # Draw x,y,z-lines through dipole position
        # mlab.plot3d([x_lim[0], dipole_pos[0], x_lim[1]],
        #             [dipole_pos[1], dipole_pos[1], dipole_pos[1]],
        #             [dipole_pos[2], dipole_pos[2], dipole_pos[2]],
        #             color=(1,0,0), tube_radius=0.5, figure=self.scene2.mayavi_scene)
        #
        # mlab.plot3d([dipole_pos[0], dipole_pos[0], dipole_pos[0]],
        #             [y_lim[0], dipole_pos[1], y_lim[1]],
        #             [dipole_pos[2], dipole_pos[2], dipole_pos[2]],
        #             color=(0,1,0), tube_radius=0.5, figure=self.scene2.mayavi_scene)
        #
        # mlab.plot3d([dipole_pos[0], dipole_pos[0], dipole_pos[0]],
        #             [dipole_pos[1], dipole_pos[1], dipole_pos[1]],
        #             [z_lim[0], dipole_pos[2], z_lim[1]],
        #             color=(0,0,1), tube_radius=0.5, figure=self.scene2.mayavi_scene)

        # Draw dipole direction
        print(self.ny_head.dipole_moment[0])
        mlab.quiver3d(np.array([self.ny_head.dipole_pos[0]]),
                      np.array([self.ny_head.dipole_pos[1]]),
                      np.array([self.ny_head.dipole_pos[2]]),
                      np.array(self.ny_head.dipole_moment[0]),
                      np.array(self.ny_head.dipole_moment[1]),
                      np.array(self.ny_head.dipole_moment[2]),
                      scale_factor=40, line_width=5,
                      color=(0,0,0), figure=self.scene2.mayavi_scene)

    # The layout of the dialog created

    view = View(HSplit(
                  Group(
                       Item('scene1',
                            editor=SceneEditor(), height=250,
                            width=300),
                       show_labels=False,
                  ),
                  Group(
                       Item('scene2',
                            editor=SceneEditor(), height=100,
                            width=100, show_label=False),
                       show_labels=False,
                  ),
                ),
                resizable=True,
                )

    def __init__(self, ny_head):
        self.ny_head = ny_head
        self.vmax = np.max(np.abs(ny_head.eeg))
        self.vmin = -self.vmax
        self.head_color = (0.95, 0.83, 0.83)
        self.draw_scene1()
        self.draw_scene2()

if __name__ == '__main__':
    from main import NYHeadModel

    ny_head = NYHeadModel()
    ny_head.set_dipole_pos('occipital_lobe')
    ny_head.set_dipole_moment(ny_head.cortex_normals[:, ny_head.closest_vertex_idx])
    ny_head.calculate_eeg_signal()
    m = MyDialog(ny_head)
    m.configure_traits()
