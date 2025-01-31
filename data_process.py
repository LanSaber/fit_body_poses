import sys

import numpy as np
import pickle

import json
import trimesh

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QFileDialog, QPushButton, QWidget
from pyqtgraph.opengl import GLViewWidget, GLMeshItem, GLGridItem, GLScatterPlotItem, MeshData

import matplotlib.pyplot as plt

def generate_colors_from_colormap(num_colors, colormap):
    colors = []
    cmap = plt.get_cmap(colormap)

    for i in np.linspace(0, 1, num_colors):
        rgba = cmap(i)
        color = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), 0)
        colors.append(color)

    return colors

with open("phoenix-2014-keypoints.pkl", "rb") as f:
    keypoints = pickle.load(f)

with open("/home/hhm/SLRT/Spoken2Sign/results/smplsignx_intp_phoenix/results/phoenix_000000_001500.pkl", "rb") as f:
    results = pickle.load(f)

img_path = "/home/hhm/phoenix/phoenix-2014.v3/phoenix2014-release/phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px"


with open("data_folder/keypoints/000000_keypoints.json") as f:
    keypoints_json = json.load(f)

mesh = trimesh.load("/home/hhm/smplify-x/output_folder/meshes/data_folder/1/01April_2010_Thursday_heute.avi_pid0_fn000000-0.png/080.obj", force="mesh")

vertex_color_array = np.zeros((10475, 4))
color_array = generate_colors_from_colormap(27, "hsv")

with open("/home/hhm/smpl_model/smplx_vert_segmentation.json", "r", encoding="utf-8") as f:
    vertices_map = json.load(f)
    for i, (key, vertex_list) in enumerate(vertices_map.items()):
        for vertex_index in vertex_list:
            vertex_color_array[vertex_index] = color_array[i]

def add_lower_body(keypoint:np.array):
    # data process
    mapping = np.array([0, 6, 6, 8, 10, 5, 7, 9, 12, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 18, 19, 21, 21, 22],
                       dtype=np.int32)
    keypoints_openpose = np.zeros([keypoint.shape[0], 118, 3])
    for k, indice in enumerate(mapping):
        keypoints_openpose[:, k, :] = keypoint[:, indice, :]
    keypoints_openpose[:, 1, 0] = (keypoint[:, 5, 0] + keypoint[:, 6, 0]) / 2
    keypoints_openpose[:, 8, 0] = (keypoint[:, 12, 0] + keypoint[:, 11, 0]) / 2
    keypoints_openpose[:, 25:67, :] = keypoint[:, 91:, :]
    keypoints_openpose[:, 67:, :] = keypoint[:, 40:91, :]

    keypoints_openpose[:, :, 0] *= 1.24

    len_shoulder = keypoints_openpose[:, 5, 0] - keypoints_openpose[:, 2, 0]
    len_waist = len_shoulder / 1.7
    keypoints_openpose[:, 8, 0] = keypoints_openpose[:, 1, 0]
    keypoints_openpose[:, 8, 1] = keypoints_openpose[:, 1, 1] + 1.5 * len_shoulder
    keypoints_openpose[:, 9, 0] = keypoints_openpose[:, 8, 0] - 0.5 * len_waist
    keypoints_openpose[:, 9, 1] = keypoints_openpose[:, 8, 1]
    keypoints_openpose[:, 12, 0] = keypoints_openpose[:, 8, 0] + 0.5 * len_waist
    keypoints_openpose[:, 12, 1] = keypoints_openpose[:, 8, 1]
    keypoints_openpose[:, 10, 0] = keypoints_openpose[:, 9, 0]
    keypoints_openpose[:, 10, 1] = keypoints_openpose[:, 9, 1] + 2. * len_waist
    keypoints_openpose[:, 11, 0] = keypoints_openpose[:, 9, 0]
    keypoints_openpose[:, 11, 1] = keypoints_openpose[:, 9, 1] + 4. * len_waist
    keypoints_openpose[:, 13, 0] = keypoints_openpose[:, 12, 0]
    keypoints_openpose[:, 13, 1] = keypoints_openpose[:, 12, 1] + 2. * len_waist
    keypoints_openpose[:, 14, 0] = keypoints_openpose[:, 12, 0]
    keypoints_openpose[:, 14, 1] = keypoints_openpose[:, 12, 1] + 4. * len_waist
    keypoints_openpose[:, 8:15, 2] = 0.65

    lower_body_array = np.zeros([keypoint.shape[0], 7, 3])
    lower_body_array = keypoints_openpose[:, 8:15, :]

    keypoint = np.concatenate((keypoint, lower_body_array), axis=1)
    keypoint[:,:133,0] = keypoint[:,:133,0] * 1.24
    return keypoint


class VisualWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.window = GLViewWidget()

        self.window.setGeometry(1000, 0, 1920, 1080)
        self.window.setCameraPosition(distance=1.5, elevation=1.8)
        self.window.setBackgroundColor(0.8)

        self.window.show()

        zgrid = GLGridItem(color=(255, 255, 255, 226))
        self.window.addItem(zgrid)
        pkl_source = False
        obj_source = True
        json_source = False
        if pkl_source == True:
            # target point
            key_list = list(keypoints.keys())
            keypoint = keypoints[key_list[0]]["keypoints"]

            keypoint = add_lower_body(keypoint)

            # keypoint[:, 20, 2] += keypoint[:, 20, 2]+10

            self.target_pointcolors = np.zeros([keypoint.shape[1], 4])
            for i in range(140):
                # upper body
                if ((i >= 0 and i < 13)):
                    self.target_pointcolors[i] = np.array([1, 1, 1, 1])
                # upper body
                elif i >= 13 and i < 91 and i != 19:
                    self.target_pointcolors[i] = np.array([0, 0, 0, 0])

                # hands
                elif i >= 91:
                    # right hands
                    if i >= 91 and i < 112:
                        self.target_pointcolors[i] = np.array([1, 1, 1, 1])

                    # left hands
                    elif i >= 112 and i < 133:
                        self.target_pointcolors[i] = np.array([1, 1, 1, 1])
                    else:
                        self.target_pointcolors[i] = np.array([1, 1, 1, 1])
                else:
                    self.target_pointcolors[i] = np.array([1, 1, 1, 0])
            # choose frame number
            points = keypoint[0]
            vertices = GLScatterPlotItem(pos=points, color=self.target_pointcolors)
            self.window.addItem(vertices)
        if obj_source == True:
            # Extract vertices and faces from the mesh
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)

            # Create the GLMeshItem
            mesh_data = MeshData(vertexes=vertices, faces=faces, vertexColors=vertex_color_array)
            mesh_item = GLMeshItem(
                meshdata = mesh_data,
                edgeColor=(0.0, 0.0, 0.0, 1.0),  # RGBA color for the edges
                drawEdges=False       # Enable drawing edges
            )
            self.window.addItem(mesh_item)

        if json_source == True:
            keypoints_openpose = keypoints_json["people"][0]["pose_keypoints_2d"]
            points = np.array(keypoints_openpose).reshape(-1, 3)
            self.target_pointcolors = np.zeros((points.shape[0],4))
            for i in range(self.target_pointcolors.shape[0]):
                self.target_pointcolors[i, :] = np.array([1, 1, 1, 0.5])
            vertices = GLScatterPlotItem(pos=points, color=self.target_pointcolors)
            self.window.addItem(vertices)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    visualWindow = VisualWindow()
    sys.exit(app.exec_())