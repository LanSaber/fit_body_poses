import sys
import trimesh
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QFileDialog, QPushButton, QWidget
from pyqtgraph.opengl import GLViewWidget, GLMeshItem


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Viewer with PyQtGraph")
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Load button
        self.load_button = QPushButton("Load .OBJ File")
        self.load_button.clicked.connect(self.load_obj_file)
        self.layout.addWidget(self.load_button)

        # 3D viewer (GLViewWidget)
        self.viewer = GLViewWidget()
        self.viewer.setCameraPosition(distance=10)
        self.layout.addWidget(self.viewer)

        # Storage for the current mesh item
        self.mesh_item = None


    def load_obj_file(self):
        """Load an .obj file and render it."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open .OBJ File", "", "OBJ Files (*.obj);;All Files (*)", options=options
        )

        if file_path:
            try:
                # Load the .obj file using trimesh
                mesh = trimesh.load(file_path, force="mesh")
                if not mesh.is_empty:
                    print(f"Successfully loaded: {file_path}")
                    self.display_mesh(mesh)
                else:
                    print("Failed to load mesh: The file may be invalid.")
            except Exception as e:
                print(f"Error loading file: {e}")

    def display_mesh(self, mesh):
        """Render the mesh in the GLViewWidget."""
        if self.mesh_item:
            self.viewer.removeItem(self.mesh_item)  # Remove existing mesh

        # Extract vertices and faces from the mesh
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        # Create the GLMeshItem
        self.mesh_item = GLMeshItem(
            vertexes=vertices,  # Vertices of the mesh
            faces=faces,        # Faces of the mesh
            faceColor=(1.0, 0.5, 0.5, 1.0),  # RGBA color for the faces
            edgeColor=(0.0, 0.0, 0.0, 1.0),  # RGBA color for the edges
            drawEdges=True       # Enable drawing edges
        )
        
        # Add the mesh to the viewer
        self.viewer.addItem(self.mesh_item)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())