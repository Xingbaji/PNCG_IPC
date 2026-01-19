import numpy as np
import taichi as ti
np.set_printoptions(suppress=True)
import meshtaichi_patcher as Patcher
from scipy.spatial.transform import Rotation
import os


def compute_auto_camera(vertices, fov_degrees=45.0, padding=1.5):
    """
    Automatically compute camera position and lookat based on mesh bounding box.

    Args:
        vertices: numpy array of shape (N, 3) containing vertex positions
        fov_degrees: camera field of view in degrees (default 45)
        padding: multiplier for camera distance to ensure entire mesh is visible (default 1.5)

    Returns:
        camera_position: list [x, y, z] for camera position
        camera_lookat: list [x, y, z] for camera lookat point (bounding box center)
    """
    # Compute bounding box
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)

    # Center of bounding box is the lookat point
    center = (bbox_min + bbox_max) / 2.0

    # Compute bounding box diagonal (maximum extent)
    bbox_size = bbox_max - bbox_min
    diagonal = np.linalg.norm(bbox_size)

    # Compute camera distance based on FOV to fit the entire object
    # Using the formula: distance = (diagonal/2) / tan(fov/2)
    fov_radians = np.radians(fov_degrees)
    distance = (diagonal / 2.0) / np.tan(fov_radians / 2.0) * padding

    # Position camera along a diagonal direction (back-right-up from center)
    # This gives a nice 3/4 view of the object
    direction = np.array([1.0, 0.5, 1.0])
    direction = direction / np.linalg.norm(direction)

    camera_position = center + direction * distance

    return camera_position.tolist(), center.tolist()

@ti.data_oriented
class model_loading:
    def __init__(self, demo):
        # Try new config system first (if available)
        if self._try_load_from_registry(demo):
            return

        # Fall back to legacy if-elif chain
        if demo == 'armadillo_collision_free':
            demo_dict = {'E': 5e4, 'nu': 0.4, 'density': 1.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 5e-5, 'iter_max': 150, 'height': 1.0, 'elastic_type': 'FCR_filter',
                         'model_paths': ['../model/mesh/Armadillo13K/Armadillo13K.node'],
                         'rotations': [[0, 0, 0]], 'scales': [[1.0, 1.0, 1.0]], 'translations': [[0.0, 0.0, 0.0]],
                         'camera_position': [73.29204366, 65.86544759, 94.53984166],
                         'camera_lookat': [72.2562863, 65.09321715, 93.25890768],
                         }
            self.load_demo_n_object_collision_free(demo, demo_dict)
        elif demo == 'banana':
            demo_dict = {'E': 5e4, 'nu': 0.3, 'density': 100.0, 'gravity': -9.8, 'dt': 0.01,
                         'epsilon': 1e-7, 'iter_max': 50, 'height': 3.0, 'elastic_type': 'ARAP_filter',
                         'model_paths': ['../model/mesh/banana/banana.node'], 'rotations': [[0, 0, 0]], 'scales': [[1.0, 1.0, 1.0]], 'translations': [[0.0, 0.0, 0.0]],
                         'camera_position': [2.0, 2.0, 3.95],
                         'camera_lookat': [0.5, 0.5, 0.5],
                         }
            self.load_demo_n_object_collision_free(demo, demo_dict)
        elif demo == 'cube':
            demo_dict = {'E': 1e4, 'nu': 0.4, 'density': 500.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-10, 'iter_max': 50, 'height': 4.0, 'elastic_type': 'ARAP_filter',
                         'model_paths': ['../model/mesh/cube/cube.node'], 'rotations': [[0, 0, 0]],
                         'scales': [[1.0, 1.0, 1.0]], 'translations': [[0.0, 0.0, 0.0]],
                         'camera_position': [0.16850185, -1.69999744, 5.03710925],
                         'camera_lookat': [0.14355508, -1.50024417, 4.05758065],
                         }
            self.load_demo_n_object_collision_free(demo, demo_dict)
        elif demo == 'cube_10':
            demo_dict = {'E': 1e4, 'nu': 0.4, 'density': 500.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-7, 'iter_max': 100,  'height': 5.0, 'elastic_type': 'ARAP_filter',
                         'model_paths': ['../model/mesh/cube_10/cube_10.node'],
                         'rotations': [[0, 0, 0]],
                         'scales': [[1.0, 1.0, 1.0]], 'translations': [[0.0, 0.0, 0.0]],
                         'camera_position': [0.44853318, 0.50973239, 0.34616475],
                         'camera_lookat': [0.4503797, 0.53796244, -0.653435],
                         }
            self.load_demo_n_object_collision_free(demo, demo_dict)
        elif demo == 'cube_20':
            demo_dict = {'E': 1e4, 'nu': 0.4, 'density': 500.0, 'gravity': -9.8, 'dt': 0.01,
                         'epsilon': 1e-5, 'iter_max': 100,  'height': 1.0, 'elastic_type': 'ARAP_filter',
                         'model_paths': ['../model/mesh/cube_20/cube_20.node'],
                         'rotations': [[90, 0, 0]],
                         'scales': [[1.0, 1.0, 1.0]], 'translations': [[0.0, 0.0, 0.0]],
                         'camera_position': [0.44853318, 0.50973239, 0.34616475],
                         'camera_lookat': [0.4503797, 0.53796244, -0.653435],
                         }
            self.load_demo_n_object_collision_free(demo, demo_dict)
        elif demo == 'cube_40':
            demo_dict = {'E': 1e4, 'nu': 0.4, 'density': 500.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-5, 'iter_max': 50,  'height': 4.0, 'elastic_type': 'SNH',
                         'model_paths': ['../model/mesh/cube_40/cube_40.node'],
                         'rotations': [[90, 0, 0]],
                         'scales': [[1.0, 1.0, 1.0]], 'translations': [[0.0, 0.0, 0.0]],
                         'camera_position': [0.4426824, 1.04414034, 1.44265061],
                         'camera_lookat': [0.45726201, 1.19416167, 0.45407536],
                         }
            self.load_demo_n_object_collision_free(demo, demo_dict)
        elif demo == 'two_E_demo_collision_free':
            demo_dict = {'E': 1e4, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
                         'epsilon': 1e-5, 'iter_max': 50,  'height': 0.1, 'elastic_type': 'SNH', 'adj': 0, 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/e_2/e_2.node' for _ in range(2)],
                         'rotations': [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                         'scales': [[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]],
                         'translations': [[0.0, 0.0, 0.0],[0.0, 1.5, 0.0]],
                         'camera_position': [2.02077697, -0.54062709, 2.59427191],
                         'camera_lookat': [1.34371885, -0.79285719, 1.90291651],
                         }
            self.load_demo_n_object(demo, demo_dict)

        elif demo == 'eight_E_drop_demo_contact':
            demo_dict = {'E': 1e4, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
                         'epsilon': 1e-4, 'iter_max': 50,  'height': 0.5,
                         'dHat': 0.025, 'kappa': 0.5, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 1,
                         'model_paths': ['../model/mesh/e_2/e_2.node' for _ in range(8)],
                         'rotations': [[0.0, 0.0, 0.0] for _ in range(8)],
                         'scales': [[1.0, 1.0, 1.0] for _ in range(8)],
                         'translations': [[1.5 * j, 1.5*i, 0.0] for i in range(4) for j in range(2)],
                         'camera_position': [2.02077697, -0.54062709, 2.59427191],
                         'camera_lookat': [1.34371885, -0.79285719, 1.90291651],
                         }
            self.load_demo_n_object(demo, demo_dict)

        elif demo == 'twist_mat150':
            demo_dict = {'E': 1e4, 'nu': 0.4, 'density': 1000.0, 'gravity': 0.0, 'dt': 0.04,
                         'epsilon': 1e-3, 'iter_max': 150, 'height': 0.0,
                         'dHat': 0.004, 'kappa': 0.5, 'elastic_type': 'FCR_filter', 'adj': 0, 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/mat150x150t40_new/mat150x150t40_new.node'],
                         'dirichlet_path': '../model/mesh/mat150x150t40_new/is_dirichlet.npy',
                         'rotations': [[90.0, 0.0, 0]], 'scales': [[1.0, 1.0, 1.0]], 'translations': [[0.0, 0.0, 0.0]],
                         'camera_position': [-0.10322845, -0.26977274, -1.4946556],
                         'camera_lookat': [-0.05233857, -0.15009736, -0.50314765],
                         }
            self.load_demo_n_object_dirichlet(demo, demo_dict)
        elif demo == 'twist_rods':
            demo_dict = {'E': 1e4, 'nu': 0.4, 'density': 1000.0, 'gravity': 0.0, 'dt': 0.04,
                         'epsilon': 1e-3, 'iter_max': 150, 'height': 0.0,
                         'dHat': 0.001, 'kappa':  0.1, 'elastic_type': 'FCR_filter', 'adj': 0, 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/rod300x33/rod300x33.node' for _ in range(4)],
                         'dirichlet_path': '../model/mesh/rod300x33/is_dirichlet.npy',
                         'rotations': [[0.0, 0.0, 0] for _ in range(4)], 'scales':  [[1.0, 1.0, 1.0] for _ in range(4)],
                         'translations': [[0.0, -0.1, -0.1],[0.0, -0.1, 0.1],[0.0, 0.1, -0.1],[0.0, 0.1, 0.1]],
                         'camera_position': [-0.09454866, -0.05231105, -0.56085817],
                         'camera_lookat': [-0.01724642, 0.03024886, 0.43272535],
                         }
            self.load_demo_n_object_dirichlet(demo, demo_dict)
        elif demo == 'squeeze_four_armadillo':
            demo_dict = {'E': 2e4, 'nu': 0.4, 'density': 16.0, 'gravity': 0.0, 'dt': 0.04,
                         'epsilon': 1e-4, 'iter_max': 30, 'height': 100.0,
                         'dHat': 1.5, 'kappa': 16.0, 'elastic_type': 'ARAP', 'adj': 1, 'ground_barrier': 1,
                         'model_paths': ['../model/mesh/Armadillo13K/Armadillo13K.node' for _ in range(4)],
                         'rotations': [[0, 0, 0] for _ in range(4)],
                         'scales': [[1.0, 1.0, 1.0] for _ in range(4)],
                         'translations': [[0.0, 0.0, 0.0], [150.0, 0.0, 0.0], [0.0, 0.0, 150.0], [150.0, 0.0, 150.0]],
                         'camera_position': [233.31213006, 128.41546321, 100.97038587],
                         'camera_lookat': [232.58192635, 127.73223546, 100.96884313],
                         }
            self.load_demo_n_object(demo, demo_dict)

        elif demo == 'four_long_noodle':
            demo_dict = {'E': 5e3, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-4, 'iter_max': 30, 'height': 0,
                         'dHat': 0.004, 'kappa': 1.0, 'elastic_type': 'ARAP_filter', 'adj': 0, 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/long_noodle_750/long_noodle_750.1.node' for _ in range(4)] + ['../model/mesh/Bowl/Bowl.2.node'],
                         'rotations': [[0.0, 0.0, 90] for _ in range(4)] + [[90.0, 0.0, 0.0]], 'scales': [[1.0, 1.0, 1.0] for _ in range(4)] + [[0.45, 0.45, 0.45]],
                         'translations': [[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.0, 0.25], [0.0, 0.0, -0.25],  [0.0, -15.1, 0.0]],
                         'camera_position': [0.53409419, -14.25542546, -0.36805832],
                         'camera_lookat': [-0.06181073, -14.54020105, -0.06550654],
                         }
            self.load_demo_n_object(demo, demo_dict)

        elif demo == 'four_long_noodle2':
            demo_dict = {'E': 1e4, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-4, 'iter_max': 60, 'height': 0.5,
                         'dHat': 0.004, 'kappa': 5.0, 'elastic_type': 'ARAP_filter', 'adj': 0, 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/long_noodle_750/long_noodle_750.1.node' for _ in range(4)],
                         'rotations': [[0.0, 0.0, 90] for _ in range(4)], 'scales': [[1.0, 1.0, 1.0] for _ in range(4)],
                         'translations': [[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.0, 0.25], [0.0, 0.0, -0.25]],
                         'camera_position': [0.53409419, -14.25542546, -0.36805832],
                         'camera_lookat': [-0.06181073, -14.54020105, -0.06550654],
                         }
            self.load_demo_n_object(demo, demo_dict)

        elif demo == 'unittest_wedge_wedge':
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-7, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.1, 'kappa': 4.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/wedge/wedge.node','../model/mesh/wedge/wedge.node'],
                         'rotations': [[0.0, 0.0, 0.0],[0.0, 0.0, 180.0]],
                         'scales': [[1.0,1.0,1.0],[1.0,1.0,1.0]],
                         'translations': [[0.0, 0.0, 0.0], [0.0, 7.0, 0.0]],
                         'camera_position': [-1.378518, 3.47533885, 5.72554384],
                         'camera_lookat': [-1.38104686, 3.32964677, 4.73621708],
                         }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'unittest_wedge_spike':
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-7, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.1, 'kappa': 5.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/wedge/wedge.node','../model/mesh/spike/spike.node'],
                         'rotations': [[0.0, 0.0, 0.0],[180.0, 0.0, 0.0]],
                         'scales': [[1.0,1.0,1.0],[1.0,1.0,1.0]],
                         'translations': [[0.0, 0.0, 0.0], [0.0, 6.5, 0.0]],
                         'camera_position': [-2.56836436, 3.19896263, 4.71955579],
                         'camera_lookat': [-2.36268801, 3.13648799, 3.74293193],
                         }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'unittest_wedge_spike_cubic':
            # Same as unittest_wedge_spike but using cubic barrier instead of log barrier
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-7, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.1, 'kappa': 5.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 0,
                         'barrier_type': 'cubic',
                         'model_paths': ['../model/mesh/wedge/wedge.node','../model/mesh/spike/spike.node'],
                         'rotations': [[0.0, 0.0, 0.0],[180.0, 0.0, 0.0]],
                         'scales': [[1.0,1.0,1.0],[1.0,1.0,1.0]],
                         'translations': [[0.0, 0.0, 0.0], [0.0, 6.5, 0.0]],
                         'camera_position': [-2.56836436, 3.19896263, 4.71955579],
                         'camera_lookat': [-2.36268801, 3.13648799, 3.74293193],
                         }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'unittest_wedge_spike_adaptive':
            # Cubic barrier with adaptive kappa (elasticity-inclusive dynamic stiffness)
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-7, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.1, 'kappa': 1.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 0,
                         'barrier_type': 'cubic', 'adaptive_kappa': True,
                         'model_paths': ['../model/mesh/wedge/wedge.node','../model/mesh/spike/spike.node'],
                         'rotations': [[0.0, 0.0, 0.0],[180.0, 0.0, 0.0]],
                         'scales': [[1.0,1.0,1.0],[1.0,1.0,1.0]],
                         'translations': [[0.0, 0.0, 0.0], [0.0, 6.5, 0.0]],
                         'camera_position': [-2.56836436, 3.19896263, 4.71955579],
                         'camera_lookat': [-2.36268801, 3.13648799, 3.74293193],
                         }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'unittest_spike_spike':
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-7, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.1, 'kappa': 4.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/spike/spike.node', '../model/mesh/spike/spike.node'],
                         'rotations': [[0.0, 0.0, 0.0], [180, 0.0, 0.0]],
                         'scales': [[1.0,1.0,1.0],[1.0,1.0,1.0]],
                         'translations': [[0.0, 0.0, 0.0], [0.0, 6.5, 0.0]],
                         'camera_position': [0.41996257, 2.24740683, -3.86870303],
                         'camera_lookat': [0.46217027, 2.22056358, -2.86995484],
                         }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'unittest_crack_spike':
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-6, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.1, 'kappa': 10.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/crack/crack.node', '../model/mesh/spike/spike.node'],
                         'rotations': [[0.0, 0.0, 0.0],[0.0, 0.0, 180.0]],
                         'scales': [[1.0,1.0,1.0],[1.0,1.0,1.0]],
                         'translations': [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
                         'camera_position': [-0.09194252, 1.33939877, 6.36510762],
                         'camera_lookat': [-0.11727148, 1.23811724, 5.3705723],
                         }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'unittest_crack_wedge':
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-6, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.1, 'kappa': 20.0, 'elastic_type': 'SNH', 'adj': 0, 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/crack/crack.node', '../model/mesh/wedge/wedge.node'],
                         'rotations': [[0.0, 0.0, 0.0],[0.0, 0.0, 180.0]],
                         'scales': [[1.0,1.0,1.0],[1.0,1.0,1.0]],
                         'translations': [[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
                         'camera_position': [1.60707535, 2.84365887, -6.085546],
                         'camera_lookat': [1.61414123, 2.51783351, -5.14014243],
                         }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'unittest_edge_spike':
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-7, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.1, 'kappa': 20.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/internal_edges/internal_edges.node','../model/mesh/spike/spike.node',],
                         'rotations': [[0.0, 0.0, 0.0],[0.0, 0.0, 180.0]],
                         'scales': [[4.0,4.0,4.0],[1.0,1.0,1.0]],
                         'translations': [[0.0, 0.0, 0.0], [0.0, 7.0, 0.0]],
                         'camera_position': [0.1192546, 2.72441612, 3.10342869],
                         'camera_lookat': [0.09392564, 2.62313459, 2.10889338],
                         }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'unittest_cube_spike2': # use large cube for silding test
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-6, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.1, 'kappa': 4.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/cube/cube.node','../model/mesh/spike/spike.node',],
                         'rotations': [[0.0, 0.0, 0.0],[0.0, 0.0, 180.0]],
                         'scales': [[8.0,8.0,8.0],[0.5,0.5,0.5]],
                         'translations': [[-1.0, 0.0, -1.0], [0.0, 8.0, 0.0]],
                         'camera_position': [2.41025122, 4.40999399, 0.11444951],
                         'camera_lookat': [1.48162284, 4.04749814, 0.03541727],
                         }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'unittest_cube_wedge':  # use large cube for silding
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-6, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.2, 'kappa': 20.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/cube/cube.node', '../model/mesh/wedge/wedge.node', ],
                         'rotations': [[0.0, 0.0, 0.0], [0.0, 0.0, 180.0]],
                         'scales': [[16.0, 16.0, 16.0], [1.0, 1.0, 1.0]],
                         'translations': [[1.0, 0.0, 1.0], [0.0, 12.0, 0.0]],
                         'camera_position': [-5.865148, 9.39438321, 0.08294428],
                         'camera_lookat': [-4.92039291, 9.1154104, -0.08913708],
                         }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'unittest_edge_cube':
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-10, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.05, 'kappa': 1.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/internal_edges/internal_edges.node', '../model/mesh/cube/cube.node'],
                         'rotations': [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                         'scales': [[1.0,1.0,1.0],[1.0,1.0,1.0]],
                         'translations': [[0.0, 0.0, 0.0], [0.0, 1.2, 0.0]],
                         'camera_position': [0.10797674, 1.26187121, 2.66060623],
                         'camera_lookat': [0.08302997, 1.16211794, 1.68107763],
                         }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'unittest_cliff_cube':
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-8, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.1, 'kappa': 1.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/cliff/cliff.node', '../model/mesh/cube/cube.node'],
                         'rotations': [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                         'scales': [[1.0,1.0,1.0],[1.0,1.0,1.0]],
                         'translations': [[0.0, 0.0, 0.0], [0.0, 1.2, 0.0]],
                         'camera_position': [0.10797674, 1.26187121, 2.66060623],
                         'camera_lookat': [0.08302997, 1.16211794, 1.68107763],
                         }
            self.load_demo_n_object(demo, demo_dict)

        # ========== Stiff-GIPC Scenes ==========
        # Scenes ported from /root/Stiff-GIPC/StiffGIPC/scene_setup.cu

        # Scene 0: Octopus Stack - 4 octopuses stacked vertically
        elif demo == 'stiff_octopus_stack':
            meshes = []
            rotations = []
            scales = []
            translations = []
            for i in range(4):
                meshes.append('../model/mesh/octopus/octopus.node')
                rotations.append([-90.0, 0.0, 0.0])  # Rotate -90 degrees around X
                scales.append([1.0, 1.0, 1.0])
                translations.append([0.0, 0.0, -0.65 + 0.3 * i])
            demo_dict = {
                'E': 1e6, 'nu': 0.4, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.1,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 1,
                'model_paths': meshes,
                'rotations': rotations,
                'scales': scales,
                'translations': translations,
            }
            self.load_demo_n_object(demo, demo_dict)

        # Scene 1: Single Bunny
        elif demo == 'stiff_single_bunny':
            demo_dict = {
                'E': 8.5e5, 'nu': 0.4, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.1,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 1,
                'model_paths': ['../model/mesh/bunny_stiff/bunny_stiff.node'],
                'rotations': [[0.0, 0.0, 0.0]],
                'scales': [[0.5, 0.5, 0.5]],
                'translations': [[0.0, 0.5, 0.0]],
            }
            self.load_demo_n_object(demo, demo_dict)

        # Scene 2: Two Bunnies
        elif demo == 'stiff_two_bunnies':
            demo_dict = {
                'E': 8.5e5, 'nu': 0.4, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.1,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 1,
                'model_paths': ['../model/mesh/bunny_stiff/bunny_stiff.node'] * 2,
                'rotations': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                'scales': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                'translations': [[0.0, 0.5, 0.0], [0.25, 1.2, 0.1]],
            }
            self.load_demo_n_object(demo, demo_dict)

        # Scene 3: Stretching Armadillo - gravity stretched with fixed vertices
        elif demo == 'stiff_stretching_armadillo':
            demo_dict = {
                'E': 2.5e5, 'nu': 0.4, 'density': 1000.0, 'gravity': -49.0, 'dt': 0.01,  # 5x gravity
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.0,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 0,
                'model_paths': ['../model/mesh/armadillo_stiff/armadillo_stiff.node'],
                'rotations': [[0.0, 0.0, 0.0]],
                'scales': [[1.0, 1.0, 1.0]],
                'translations': [[0.0, 0.0, 0.0]],
            }
            self.load_demo_n_object(demo, demo_dict)

        # Scene 4: Twisting Mat - requires Dirichlet BCs
        elif demo == 'stiff_twisting_mat':
            demo_dict = {
                'E': 8.5e5, 'nu': 0.4, 'density': 1000.0, 'gravity': 0.0, 'dt': 0.01,
                'epsilon': 1e-3, 'iter_max': 150, 'height': 0.0,
                'dHat': 0.005, 'kappa': 1.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 0,
                'model_paths': ['../model/mesh/mat/mat.node'],
                'rotations': [[0.0, 0.0, 0.0]],
                'scales': [[1.0, 1.0, 1.0]],
                'translations': [[0.0, 0.0, 0.0]],
            }
            self.load_demo_n_object(demo, demo_dict)

        # Scene 6: Dragon High - large dragon mesh
        elif demo == 'stiff_dragon_high':
            demo_dict = {
                'E': 8.5e5, 'nu': 0.4, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.005,
                'epsilon': 1e-3, 'iter_max': 100, 'height': 0.1,
                'dHat': 0.001, 'kappa': 1.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 1,
                'model_paths': ['../model/mesh/dragon_50k/dragon_50k.node'],
                'rotations': [[0.0, 0.0, 0.0]],
                'scales': [[0.1, 0.1, 0.1]],
                'translations': [[0.0, 0.0, 0.0]],
            }
            self.load_demo_n_object(demo, demo_dict)

        # Scene 7: Box Pipe - Grid of cubes (simplified: 4x4 instead of 4x4x4 per layer)
        elif demo == 'stiff_box_pipe':
            meshes = []
            rotations = []
            scales = []
            translations = []
            dist = 0.2
            count = 4
            # Create 4x4 grid of cubes at different heights
            for layer in range(2):
                height = 0.1 + layer * 0.5
                for i in range(count):
                    for j in range(count):
                        meshes.append('../model/mesh/cube_stiff/cube_stiff.node')
                        rotations.append([0.0, 0.0, 0.0])
                        scales.append([0.4, 0.4, 0.4])
                        x = (i - (count-1)/2) * dist
                        z = (j - (count-1)/2) * dist
                        translations.append([x, height, z])
            demo_dict = {
                'E': 1e4, 'nu': 0.4, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.0,
                'dHat': 0.02, 'kappa': 1.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 1,
                'model_paths': meshes,
                'rotations': rotations,
                'scales': scales,
                'translations': translations,
            }
            self.load_demo_n_object(demo, demo_dict)

        # Scene 11: Two Cubes Drop - stiff cube + soft cube
        elif demo == 'stiff_two_cubes_drop':
            demo_dict = {
                'E': 1e4, 'nu': 0.4, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.0,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 1,
                'model_paths': ['../model/mesh/cube_stiff/cube_stiff.node'] * 2,
                'rotations': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                'scales': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                'translations': [[0.0, 0.1, 0.0], [0.0, 0.6, 0.0]],
            }
            self.load_demo_n_object(demo, demo_dict)

        # Scene 12: Dropping Letters - 9 letters with increasing stiffness
        elif demo == 'stiff_dropping_letters':
            meshes = []
            rotations = []
            scales = []
            translations = []
            for i in range(9):
                meshes.append('../model/mesh/letters/letters.node')
                rotations.append([0.0, 0.0, 0.0])
                scales.append([0.44, 0.44, 0.44])
                translations.append([0.017, -0.7 + i * 0.35, 0.0])
            demo_dict = {
                'E': 1e4, 'nu': 0.4, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.005,
                'epsilon': 1e-3, 'iter_max': 100, 'height': 0.0,
                'dHat': 0.002, 'kappa': 1.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 1,
                'model_paths': meshes,
                'rotations': rotations,
                'scales': scales,
                'translations': translations,
            }
            self.load_demo_n_object(demo, demo_dict)

        # Scene 13: Teapots - 32 teapots in a box (simplified: 8 teapots)
        elif demo == 'stiff_teapots':
            meshes = ['../model/mesh/box/box.node']  # Box container
            rotations = [[-90.0, 0.0, 0.0]]
            scales = [[4.0, 4.0, 4.0]]
            translations = [[0.0, 0.0, 0.0]]
            # Add teapots in a 2x4 grid
            h_space = 0.6
            x_space = 0.35
            for i in range(8):
                meshes.append('../model/mesh/teapot/teapot.node')
                rotations.append([-90.0, 0.0, 0.0])
                scales.append([1.0, 1.0, 1.0])
                row = i // 2
                col = i % 2
                x = (col - 0.5) * x_space
                y = (row - 1.5) * 0.3
                z = 0.3 + row * h_space * 0.3
                translations.append([x, y, z])
            demo_dict = {
                'E': 3e5, 'nu': 0.4, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.005,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.0,
                'dHat': 0.001, 'kappa': 1.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 1,
                'model_paths': meshes,
                'rotations': rotations,
                'scales': scales,
                'translations': translations,
            }
            self.load_demo_n_object(demo, demo_dict)

        # Scene 14: Two Soft Bunnies
        elif demo == 'stiff_two_soft_bunnies':
            demo_dict = {
                'E': 1e4, 'nu': 0.4, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.1,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'NH', 'adj': 0, 'ground_barrier': 1,
                'model_paths': ['../model/mesh/bunny_stiff/bunny_stiff.node'] * 2,
                'rotations': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                'scales': [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
                'translations': [[0.0, 0.5, 0.0], [0.0, -0.65, 0.0]],
            }
            self.load_demo_n_object(demo, demo_dict)

        # MAS Stiffness Test: 8 E objects with varying stiffness (1e4 to 1e7)
        elif demo == 'eight_E_stiffness_test':
            demo_dict = {
                'E': 1e7, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 200, 'height': 0.5,
                'dHat': 0.025, 'kappa': 0.5, 'elastic_type': 'SNH', 'adj': 0, 'ground_barrier': 1,
                'model_paths': ['../model/mesh/e_2/e_2.node' for _ in range(8)],
                'rotations': [[0.0, 0.0, 0.0] for _ in range(8)],
                'scales': [[1.0, 1.0, 1.0] for _ in range(8)],
                'translations': [[1.5 * j, 1.5 * i, 0.0] for i in range(4) for j in range(2)],
                'camera_position': [2.02077697, -0.54062709, 2.59427191],
                'camera_lookat': [1.34371885, -0.79285719, 1.90291651],
                'use_mas': False,  # Enable MAS preconditioner
            }
            self.load_demo_n_object(demo, demo_dict)
        # MAS Stiffness Test: 8 E objects with varying stiffness (1e4 to 1e7)
        elif demo == 'eight_E_stiffness_mas':
            demo_dict = {
                'E': 1e7, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 200, 'height': 0.5,
                'dHat': 0.025, 'kappa': 0.5, 'elastic_type': 'SNH', 'adj': 0, 'ground_barrier': 1,
                'model_paths': ['../model/mesh/e_2/e_2.node' for _ in range(8)],
                'rotations': [[0.0, 0.0, 0.0] for _ in range(8)],
                'scales': [[1.0, 1.0, 1.0] for _ in range(8)],
                'translations': [[1.5 * j, 1.5 * i, 0.0] for i in range(4) for j in range(2)],
                'camera_position': [2.02077697, -0.54062709, 2.59427191],
                'camera_lookat': [1.34371885, -0.79285719, 1.90291651],
                'use_mas': True,  # Enable MAS preconditioner
            }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'eight_E_cubic':
            demo_dict = {
                'E': 1e4, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.5,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'SNH', 'adj': 0, 'ground_barrier': 1,
                'barrier_type': 'cubic',  # Use cubic barrier function
                'adaptive_kappa': True,   # Use elasticity-inclusive dynamic stiffness (Eq. 4)
                'cache_kappa': True,      # Cache kappa at iter 0, reuse in subsequent iterations
                'model_paths': ['../model/mesh/e_2/e_2.node' for _ in range(8)],
                'rotations': [[0.0, 0.0, 0.0] for _ in range(8)],
                'scales': [[1.0, 1.0, 1.0] for _ in range(8)],
                'translations': [[1.5 * j, 1.5 * i, 0.0] for i in range(4) for j in range(2)],
                'camera_position': [2.02077697, -0.54062709, 2.59427191],
                'camera_lookat': [1.34371885, -0.79285719, 1.90291651],
                'use_mas': False,
            }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'eight_E_cubic_no_cache':
            # Same as eight_E_cubic but without kappa caching (update kappa every iteration)
            demo_dict = {
                'E': 1e4, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.5,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'SNH', 'adj': 0, 'ground_barrier': 1,
                'barrier_type': 'cubic',  # Use cubic barrier function
                'adaptive_kappa': True,   # Use elasticity-inclusive dynamic stiffness (Eq. 4)
                'cache_kappa': False,     # Update kappa every iteration (no caching)
                'model_paths': ['../model/mesh/e_2/e_2.node' for _ in range(8)],
                'rotations': [[0.0, 0.0, 0.0] for _ in range(8)],
                'scales': [[1.0, 1.0, 1.0] for _ in range(8)],
                'translations': [[1.5 * j, 1.5 * i, 0.0] for i in range(4) for j in range(2)],
                'camera_position': [2.02077697, -0.54062709, 2.59427191],
                'camera_lookat': [1.34371885, -0.79285719, 1.90291651],
                'use_mas': False,
            }
            self.load_demo_n_object(demo, demo_dict)            
        else:
            raise Exception('demo not found')

    def _try_load_from_registry(self, demo: str) -> bool:
        """
        Try to load configuration from the new config registry.

        Returns True if successfully loaded, False to fall back to legacy.
        """
        try:
            from config import DemoRegistry
            if not DemoRegistry.exists(demo):
                return False

            config = DemoRegistry.get(demo)
            demo_dict = config.to_legacy_dict()
            self.set_para(demo_dict)
            self.dict = demo_dict

            # Determine which load method to use based on config
            if config.ipc.enabled:
                if config.dirichlet_path:
                    self.load_demo_n_object_dirichlet(demo, demo_dict)
                else:
                    self.load_demo_n_object(demo, demo_dict)
            else:
                self.load_demo_n_object_collision_free(demo, demo_dict)

            return True
        except ImportError:
            # Config module not yet available, fall back to legacy
            return False

    def set_para(self,demo_dict):
        E = demo_dict['E']
        nu = demo_dict['nu']
        self.mu, self.la =  E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.density = demo_dict['density']
        self.dt = demo_dict['dt']
        self.gravity = demo_dict['gravity']
        self.epsilon = demo_dict['epsilon']
        self.iter_max = demo_dict['iter_max']
        self.elastic_type = demo_dict['elastic_type']
        if 'diag3x3' in demo_dict:
            self.diag3x3 = int(demo_dict['diag3x3'])
        else:
            self.diag3x3 = 0
        if 'kappa' in demo_dict:
            self.kappa = demo_dict['kappa']
            self.dHat = demo_dict['dHat']
            self.adj = int(demo_dict['adj'])
            self.ground_barrier = int(demo_dict['ground_barrier'])
            # Barrier type: 'log' (default) or 'cubic'
            if 'barrier_type' in demo_dict:
                self.barrier_type = demo_dict['barrier_type']
            else:
                self.barrier_type = 'log'
            # Adaptive kappa: dynamically computed per-constraint kappa (cubic barrier only)
            if 'adaptive_kappa' in demo_dict:
                self.adaptive_kappa = demo_dict['adaptive_kappa']
            else:
                self.adaptive_kappa = False

        # MAS preconditioner options
        if 'use_mas' in demo_dict:
            self.use_mas = demo_dict['use_mas']
        else:
            self.use_mas = False

        # METIS-based node reordering for MAS (CEMAS)
        if 'use_metis' in demo_dict:
            self.use_metis = demo_dict['use_metis']
        else:
            self.use_metis = False

    def add_object(self, model_path, translation=[0., 0., 0.], rotation=[0., 0., 0.], scale=[1, 1, 1]):
        model = Patcher.load_mesh_rawdata(model_path)
        if scale != [1., 1., 1.]:
            S = np.array([
                [scale[0], 0, 0],
                [0, scale[1], 0],
                [0, 0, scale[2]]
            ])
            model[0] = np.dot(model[0], S)
        if rotation != [0.,0.,0.]:
            rotation = np.asarray(rotation)
            rotation = rotation * np.pi / 180.0
            rotation_matrix = Rotation.from_rotvec(rotation).as_matrix()
            model[0] = np.dot(model[0], rotation_matrix)
        if translation != [0.,0.,0.]:
            model[0][:, 0] = model[0][:, 0] + translation[0]
            model[0][:, 1] = model[0][:, 1] + translation[1]
            model[0][:, 2] = model[0][:, 2] + translation[2]
        return model

    def load_demo_n_object_collision_free(self, demo, demo_dict):
        # set physical paramters
        self.set_para(demo_dict)
        self.dict = demo_dict
        models = []
        print('add model0')
        number = len(demo_dict['scales'])
        for i in range(number):
            model_i = self.add_object(model_path=demo_dict['model_paths'][i], scale=demo_dict['scales'][i], translation=demo_dict['translations'][i], rotation=demo_dict['rotations'][i])
            models.append(model_i)
            if i == 0:
                self.ground = np.min(model_i[0][:,1]) - demo_dict['height']
        self.mesh = Patcher.load_mesh(models, relations=["CV"])
        # Auto-compute camera position and lookat
        self.auto_camera_from_models(models)
        print('load finish')

    def load_demo_n_object(self, demo, demo_dict):
        # set physical paramters
        self.set_para(demo_dict)
        self.dict = demo_dict
        models = []
        print('add model0')
        number = len(demo_dict['scales'])
        ground_min = 100.0
        for i in range(number):
            model_i = self.add_object(model_path=demo_dict['model_paths'][i], scale=demo_dict['scales'][i], translation=demo_dict['translations'][i], rotation=demo_dict['rotations'][i])
            models.append(model_i)
            # print('E high ', np.max(model_i[0][:,1]) - np.min(model_i[0][:,1]))
            ground_min = min(ground_min, np.min(model_i[0][:,1]))

        self.ground = ground_min - demo_dict['height']
        print('load mesh')
        self.load_mesh_and_boundarys(demo, models)
        # Auto-compute camera position and lookat
        self.auto_camera_from_models(models)

    def load_demo_n_object_dirichlet(self, demo, demo_dict):
        # set physical paramters
        self.set_para(demo_dict)
        self.dict = demo_dict
        models = []
        print('add model0')
        number = len(demo_dict['scales'])
        ground_min = 100.0
        for i in range(number):
            model_i = self.add_object(model_path=demo_dict['model_paths'][i], scale=demo_dict['scales'][i], translation=demo_dict['translations'][i], rotation=demo_dict['rotations'][i])
            models.append(model_i)
            ground_min = min(ground_min, np.min(model_i[0][:,1]))

        self.ground = ground_min - demo_dict['height']
        print('load mesh')
        self.load_mesh_and_boundarys(demo, models)
        # Auto-compute camera position and lookat
        self.auto_camera_from_models(models)
        self.mesh.verts.place({'is_dirichlet': ti.i32})
        dirichlet_path = demo_dict['dirichlet_path']
        dirichlet_np = np.load(dirichlet_path)
        self.mesh.verts.is_dirichlet.from_numpy(dirichlet_np)

    def load_mesh_and_boundarys(self, demo, models):
        save_path = '../demo_results/final/' + demo + '/boundary/'
        if not os.path.exists(save_path + '/boundary_points.npy'):
            self.load_and_save_boundarys(demo, models)

        self.mesh = Patcher.load_mesh(models, relations=["CV"])
        boundary_points_np = np.load(save_path + '/boundary_points.npy')
        boundary_edges_np = np.load(save_path + '/boundary_edges.npy')
        boundary_triangles_np = np.load(save_path + '/boundary_triangles.npy')
        n_boundary_points = boundary_points_np.shape[0]
        n_boundary_edges = boundary_edges_np.shape[0]
        n_boundary_triangles = boundary_triangles_np.shape[0]
        print('load n_boundary_points: ', n_boundary_points, 'n_boundary_edges: ', n_boundary_edges, 'n_boundary_triangles: ', n_boundary_triangles)
        self.boundary_points = ti.field(ti.i32)
        self.boundary_edges = ti.field(ti.i32)
        self.boundary_triangles = ti.field(ti.i32)
        ti.root.dense(ti.i, n_boundary_points).place(self.boundary_points)
        print('new boundary ')

        ti.root.dense(ti.ij, (n_boundary_edges, 2)).place(self.boundary_edges)
        ti.root.dense(ti.ij, (n_boundary_triangles, 3)).place(self.boundary_triangles)
        self.boundary_points.from_numpy(boundary_points_np)
        self.boundary_edges.from_numpy(boundary_edges_np)
        self.boundary_triangles.from_numpy(boundary_triangles_np)
        print('load finish')

    def load_and_save_boundarys(self, demo,  models):
        # load demo and save boundary point ,edge, triangle of demo in the path
        # 因为meshtaichi无法在并行ele的时候还只并行surface mesh，遍历所有的surface edge 再判断是否是boundary的速度会很慢，这里把surface的信息保存下来
        save_path = '../demo_results/final/' + demo + '/boundary/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print('assign boundarys... ')
        self.mesh_tmp = Patcher.load_mesh(models, relations=['FC','FE','FV','EV'])
        self.mesh_tmp.faces.place({'is_boundary': ti.i32})
        self.mesh_tmp.edges.place({'is_boundary': ti.i32})
        self.mesh_tmp.verts.place({'is_boundary': ti.i32})
        n_points = len(self.mesh_tmp.verts)
        n_edges = len(self.mesh_tmp.edges)
        n_triangles = len(self.mesh_tmp.faces)
        print('n_points', n_points, 'n_edges', n_edges, 'n_triangles', n_triangles)
        self.find_boundarys_tmp()
        self.edges = ti.field(ti.i32)
        self.triangles = ti.field(ti.i32)
        ti.root.dense(ti.ij, (n_edges, 2)).place(self.edges)
        ti.root.dense(ti.ij, (n_triangles, 3)).place(self.triangles)
        self.assign_relations()
        edges_np = self.edges.to_numpy()
        triangles_np = self.triangles.to_numpy()
        point_boundary_id = self.mesh_tmp.verts.is_boundary.to_numpy()
        point_ids = [i for i in range(n_points) if point_boundary_id[i] == 1]

        edge_boundary_id = self.mesh_tmp.edges.is_boundary.to_numpy()
        edge_ids = [i for i in range(n_edges) if edge_boundary_id[i] == 1]
        triangle_boundary_id = self.mesh_tmp.faces.is_boundary.to_numpy()
        triangle_ids = [i for i in range(n_triangles) if triangle_boundary_id[i] == 1]
        boundary_points = np.asarray(point_ids)
        boundary_edges = edges_np[edge_ids]
        boundary_triangles = triangles_np[triangle_ids]
        print('save boundary points...')
        np.save(save_path + 'boundary_points.npy', boundary_points)
        print('save boundary edges...')
        np.save(save_path + 'boundary_edges.npy', boundary_edges)
        print('save boundary triangles...')
        np.save(save_path + 'boundary_triangles.npy', boundary_triangles)
        print('boundary size', boundary_points.shape, boundary_edges.shape, boundary_triangles.shape)
        del self.mesh_tmp, self.edges, self.triangles

    @ti.kernel
    def find_boundarys_tmp(self):
        for f in self.mesh_tmp.faces:
            if f.cells.size == 1:
                f.is_boundary = 1
                f.edges[0].is_boundary = 1
                f.edges[1].is_boundary = 1
                f.edges[2].is_boundary = 1
                f.verts[0].is_boundary = 1
                f.verts[1].is_boundary = 1
                f.verts[2].is_boundary = 1

    @ti.kernel
    def assign_relations(self):
        for f in self.mesh_tmp.faces:
            id = f.id
            self.triangles[id,0] = f.verts[0].id
            self.triangles[id,1] = f.verts[1].id
            self.triangles[id,2] = f.verts[2].id
        for e in self.mesh_tmp.edges:
            id = e.id
            self.edges[id,0] = e.verts[0].id
            self.edges[id,1] = e.verts[1].id

    def auto_camera_from_models(self, models, fov_degrees=45.0, padding=1.5):
        """
        Compute camera position and lookat from loaded models.
        Only computes if camera_position/camera_lookat not already set in demo_dict.

        Args:
            models: list of model data, each model is [vertices, cells]
            fov_degrees: camera field of view in degrees
            padding: multiplier for camera distance

        Returns:
            Sets self.camera_position and self.camera_lookat
        """
        # Use manual camera settings from demo_dict if available
        if 'camera_position' in self.dict and 'camera_lookat' in self.dict:
            self.camera_position = self.dict['camera_position']
            self.camera_lookat = self.dict['camera_lookat']
        else:
            # Auto-compute camera from mesh bounding box
            all_vertices = np.vstack([model[0] for model in models])
            self.camera_position, self.camera_lookat = compute_auto_camera(
                all_vertices, fov_degrees, padding
            )
