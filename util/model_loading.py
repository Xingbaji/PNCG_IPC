"""
Model loading module for PNCG_IPC simulations.

This module provides the model_loading class that loads demo configurations
and prepares mesh data for simulation.

Configuration:
- All configurations are loaded from YAML files in demo_settings/
- Use `from demo_settings import list_demos` to see available demos

METIS Reordering:
- All meshes are automatically METIS-reordered during loading
- This provides optimal vertex ordering for MAS preconditioner
- Vertex IDs directly map to block/lane: block_id = vid // 16, lane_id = vid % 16
"""

import numpy as np
import taichi as ti
np.set_printoptions(suppress=True)
import meshtaichi_patcher as Patcher
from scipy.spatial.transform import Rotation
import os

# Import METIS reordering utilities
try:
    from algorithm.mas_preconditioner_small import (
        reorder_mesh_data_metis,
        check_pymetis_available,
        BANKSIZE,
    )
    _METIS_AVAILABLE = check_pymetis_available()
except ImportError:
    _METIS_AVAILABLE = False
    BANKSIZE = 16

    def reorder_mesh_data_metis(vertices, cells, block_size=16):
        """Fallback: return data unchanged if METIS not available."""
        return vertices, cells, None


def _merge_and_reorder_models(models, use_metis=True):
    """
    Merge multiple models and apply METIS reordering.

    Args:
        models: List of model data from add_object (each is a list: [vertices, ...cells...])
        use_metis: Whether to apply METIS reordering (default: True)

    Returns:
        Tuple of (reordered_models_dict, metis_result)
        - reordered_models_dict: Dict format for Patcher.load_mesh
        - metis_result: MetisReorderResult or None
    """
    if not models:
        return {}, None

    # Extract vertices (index 0) and cells (index 3) from each model
    all_vertices = []
    all_cells = []
    vertex_offset = 0

    for model in models:
        verts = model[0]  # vertices
        cells = model[3]  # tetrahedral cells (4 vertices per cell)
        all_vertices.append(verts)
        all_cells.append(cells + vertex_offset)
        vertex_offset += len(verts)

    merged_vertices = np.vstack(all_vertices)
    merged_cells = np.vstack(all_cells).astype(np.int32)

    # Apply METIS reordering if available
    metis_result = None
    if use_metis and _METIS_AVAILABLE:
        print(f"[model_loading] Applying METIS reordering to {len(merged_vertices)} vertices...")
        reordered_verts, reordered_cells, metis_result = reorder_mesh_data_metis(
            merged_vertices, merged_cells, BANKSIZE
        )
    else:
        if use_metis and not _METIS_AVAILABLE:
            print("[model_loading] WARNING: METIS not available, using original vertex ordering")
        reordered_verts = merged_vertices
        reordered_cells = merged_cells

    # Create dict format for Patcher.load_mesh
    reordered_dict = {0: reordered_verts, 3: reordered_cells}

    return reordered_dict, metis_result


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
    fov_radians = np.radians(fov_degrees)
    distance = (diagonal / 2.0) / np.tan(fov_radians / 2.0) * padding

    # Position camera along a diagonal direction (back-right-up from center)
    direction = np.array([1.0, 0.5, 1.0])
    direction = direction / np.linalg.norm(direction)

    camera_position = center + direction * distance

    return camera_position.tolist(), center.tolist()


@ti.data_oriented
class model_loading:
    """
    Demo configuration loader and mesh initializer.

    Usage:
        model = model_loading(demo='cube_40')
        # Access properties:
        model.dict        # Full configuration dict
        model.mu, model.la  # Lame parameters
        model.mesh        # Loaded mesh data
    """

    def __init__(self, demo):
        """
        Initialize model loading with demo name.

        Args:
            demo: Name of the demo configuration to load.
                  Must be a valid YAML config in demo_settings/.
        """
        # Load from YAML config system
        if self._try_load_from_yaml(demo):
            return

        # Try registry-based config system (for programmatic configs)
        if self._try_load_from_registry(demo):
            return

        # No config found - raise helpful error
        self._raise_demo_not_found(demo)

    def _try_load_from_yaml(self, demo: str) -> bool:
        """
        Try to load configuration from YAML files in demo_settings/.

        Returns True if successfully loaded, False to try next method.
        """
        try:
            from demo_settings import load_demo_config
            config = load_demo_config(demo)
            demo_dict = config.to_legacy_dict()
            self._load_with_dict(demo, demo_dict, config)
            return True
        except (ImportError, FileNotFoundError):
            return False

    def _try_load_from_registry(self, demo: str) -> bool:
        """
        Try to load configuration from the config registry.

        Returns True if successfully loaded, False to try next method.
        """
        try:
            from config import DemoRegistry
            if not DemoRegistry.exists(demo):
                return False

            config = DemoRegistry.get(demo)
            demo_dict = config.to_legacy_dict()
            self._load_with_dict(demo, demo_dict, config)
            return True
        except ImportError:
            return False

    def _load_with_dict(self, demo: str, demo_dict: dict, config=None):
        """
        Load demo using the configuration dict.

        Determines the appropriate load method based on config.
        """
        self.set_para(demo_dict)
        self.dict = demo_dict

        # Determine load method
        has_ipc = 'kappa' in demo_dict
        has_dirichlet = 'dirichlet_path' in demo_dict

        if has_ipc:
            if has_dirichlet:
                self.load_demo_n_object_dirichlet(demo, demo_dict)
            else:
                self.load_demo_n_object(demo, demo_dict)
        else:
            self.load_demo_n_object_collision_free(demo, demo_dict)

    def _load_legacy_config(self, demo: str):
        """Load from legacy hardcoded configurations."""
        # ============================================================
        # LEGACY CONFIGS - For backward compatibility only
        # New demos should be added as YAML files in demo_settings/
        # ============================================================

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
                         'epsilon': 1e-5, 'iter_max': 50,  'height': 0.1, 'elastic_type': 'SNH', 'ground_barrier': 0,
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
                         'dHat': 0.025, 'kappa': 0.5, 'elastic_type': 'NH', 'ground_barrier': 1,
                         'barrier_type': 'log',
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
                         'dHat': 0.004, 'kappa': 0.5, 'elastic_type': 'FCR_filter', 'ground_barrier': 0,
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
                         'dHat': 0.001, 'kappa':  0.1, 'elastic_type': 'FCR_filter', 'ground_barrier': 0,
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
                         'dHat': 1.5, 'kappa': 16.0, 'elastic_type': 'ARAP', 'ground_barrier': 1,
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
                         'dHat': 0.004, 'kappa': 1.0, 'elastic_type': 'ARAP_filter', 'ground_barrier': 0,
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
                         'dHat': 0.004, 'kappa': 5.0, 'elastic_type': 'ARAP_filter', 'ground_barrier': 0,
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
                         'dHat': 0.1, 'kappa': 4.0, 'elastic_type': 'NH', 'ground_barrier': 0,
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
                         'dHat': 0.1, 'kappa': 5.0, 'elastic_type': 'NH', 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/wedge/wedge.node','../model/mesh/spike/spike.node'],
                         'rotations': [[0.0, 0.0, 0.0],[180.0, 0.0, 0.0]],
                         'scales': [[1.0,1.0,1.0],[1.0,1.0,1.0]],
                         'translations': [[0.0, 0.0, 0.0], [0.0, 6.5, 0.0]],
                         'camera_position': [-2.56836436, 3.19896263, 4.71955579],
                         'camera_lookat': [-2.36268801, 3.13648799, 3.74293193],
                         }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'unittest_wedge_spike_cubic':
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-7, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.1, 'kappa': 5.0, 'elastic_type': 'NH', 'ground_barrier': 0,
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
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-7, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.1, 'kappa': 1.0, 'elastic_type': 'NH', 'ground_barrier': 0,
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
                         'dHat': 0.1, 'kappa': 4.0, 'elastic_type': 'NH', 'ground_barrier': 0,
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
                         'dHat': 0.1, 'kappa': 10.0, 'elastic_type': 'NH', 'ground_barrier': 0,
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
                         'dHat': 0.1, 'kappa': 20.0, 'elastic_type': 'SNH', 'ground_barrier': 0,
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
                         'dHat': 0.1, 'kappa': 20.0, 'elastic_type': 'NH', 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/internal_edges/internal_edges.node','../model/mesh/spike/spike.node',],
                         'rotations': [[0.0, 0.0, 0.0],[0.0, 0.0, 180.0]],
                         'scales': [[4.0,4.0,4.0],[1.0,1.0,1.0]],
                         'translations': [[0.0, 0.0, 0.0], [0.0, 7.0, 0.0]],
                         'camera_position': [0.1192546, 2.72441612, 3.10342869],
                         'camera_lookat': [0.09392564, 2.62313459, 2.10889338],
                         }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'unittest_cube_spike2':
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-6, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.1, 'kappa': 4.0, 'elastic_type': 'NH', 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/cube/cube.node','../model/mesh/spike/spike.node',],
                         'rotations': [[0.0, 0.0, 0.0],[0.0, 0.0, 180.0]],
                         'scales': [[8.0,8.0,8.0],[0.5,0.5,0.5]],
                         'translations': [[-1.0, 0.0, -1.0], [0.0, 8.0, 0.0]],
                         'camera_position': [2.41025122, 4.40999399, 0.11444951],
                         'camera_lookat': [1.48162284, 4.04749814, 0.03541727],
                         }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'unittest_cube_wedge':
            demo_dict = {'E': 1e5, 'nu': 0.4, 'density': 100.0, 'gravity': -9.8, 'dt': 0.04,
                         'epsilon': 1e-6, 'iter_max': 100, 'height': 0.0,
                         'dHat': 0.2, 'kappa': 20.0, 'elastic_type': 'NH', 'ground_barrier': 0,
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
                         'dHat': 0.05, 'kappa': 1.0, 'elastic_type': 'NH', 'ground_barrier': 0,
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
                         'dHat': 0.1, 'kappa': 1.0, 'elastic_type': 'NH', 'ground_barrier': 0,
                         'model_paths': ['../model/mesh/cliff/cliff.node', '../model/mesh/cube/cube.node'],
                         'rotations': [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                         'scales': [[1.0,1.0,1.0],[1.0,1.0,1.0]],
                         'translations': [[0.0, 0.0, 0.0], [0.0, 1.2, 0.0]],
                         'camera_position': [0.10797674, 1.26187121, 2.66060623],
                         'camera_lookat': [0.08302997, 1.16211794, 1.68107763],
                         }
            self.load_demo_n_object(demo, demo_dict)

        # ========== Stiff-GIPC Scenes ==========
        elif demo == 'stiff_octopus_stack':
            meshes = []
            rotations = []
            scales = []
            translations = []
            for i in range(4):
                meshes.append('../model/mesh/octopus/octopus.node')
                rotations.append([-90.0, 0.0, 0.0])
                scales.append([1.0, 1.0, 1.0])
                translations.append([0.0, 0.0, -0.65 + 0.3 * i])
            demo_dict = {
                'E': 1e6, 'nu': 0.4, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.1,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'NH', 'ground_barrier': 1,
                'model_paths': meshes,
                'rotations': rotations,
                'scales': scales,
                'translations': translations,
            }
            self.load_demo_n_object(demo, demo_dict)

        elif demo == 'stiff_single_bunny':
            demo_dict = {
                'E': 8.5e5, 'nu': 0.4, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.1,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'NH', 'ground_barrier': 1,
                'model_paths': ['../model/mesh/bunny_stiff/bunny_stiff.node'],
                'rotations': [[0.0, 0.0, 0.0]],
                'scales': [[0.5, 0.5, 0.5]],
                'translations': [[0.0, 0.5, 0.0]],
            }
            self.load_demo_n_object(demo, demo_dict)

        elif demo == 'stiff_two_bunnies':
            demo_dict = {
                'E': 8.5e5, 'nu': 0.4, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.1,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'NH', 'ground_barrier': 1,
                'model_paths': ['../model/mesh/bunny_stiff/bunny_stiff.node'] * 2,
                'rotations': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                'scales': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                'translations': [[0.0, 0.5, 0.0], [0.25, 1.2, 0.1]],
            }
            self.load_demo_n_object(demo, demo_dict)

        elif demo == 'stiff_stretching_armadillo':
            demo_dict = {
                'E': 2.5e5, 'nu': 0.4, 'density': 1000.0, 'gravity': -49.0, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.0,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'NH', 'ground_barrier': 0,
                'model_paths': ['../model/mesh/armadillo_stiff/armadillo_stiff.node'],
                'rotations': [[0.0, 0.0, 0.0]],
                'scales': [[1.0, 1.0, 1.0]],
                'translations': [[0.0, 0.0, 0.0]],
            }
            self.load_demo_n_object(demo, demo_dict)

        elif demo == 'stiff_twisting_mat':
            demo_dict = {
                'E': 8.5e5, 'nu': 0.4, 'density': 1000.0, 'gravity': 0.0, 'dt': 0.01,
                'epsilon': 1e-3, 'iter_max': 150, 'height': 0.0,
                'dHat': 0.005, 'kappa': 1.0, 'elastic_type': 'NH', 'ground_barrier': 0,
                'model_paths': ['../model/mesh/mat/mat.node'],
                'rotations': [[0.0, 0.0, 0.0]],
                'scales': [[1.0, 1.0, 1.0]],
                'translations': [[0.0, 0.0, 0.0]],
            }
            self.load_demo_n_object(demo, demo_dict)

        elif demo == 'stiff_dragon_high':
            demo_dict = {
                'E': 8.5e5, 'nu': 0.4, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.005,
                'epsilon': 1e-3, 'iter_max': 100, 'height': 0.1,
                'dHat': 0.001, 'kappa': 1.0, 'elastic_type': 'NH', 'ground_barrier': 1,
                'model_paths': ['../model/mesh/dragon_50k/dragon_50k.node'],
                'rotations': [[0.0, 0.0, 0.0]],
                'scales': [[0.1, 0.1, 0.1]],
                'translations': [[0.0, 0.0, 0.0]],
            }
            self.load_demo_n_object(demo, demo_dict)

        elif demo == 'stiff_box_pipe':
            meshes = []
            rotations = []
            scales = []
            translations = []
            dist = 0.2
            count = 4
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
                'dHat': 0.02, 'kappa': 1.0, 'elastic_type': 'NH', 'ground_barrier': 1,
                'model_paths': meshes,
                'rotations': rotations,
                'scales': scales,
                'translations': translations,
            }
            self.load_demo_n_object(demo, demo_dict)

        elif demo == 'stiff_two_cubes_drop':
            demo_dict = {
                'E': 1e4, 'nu': 0.4, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.0,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'NH', 'ground_barrier': 1,
                'model_paths': ['../model/mesh/cube_stiff/cube_stiff.node'] * 2,
                'rotations': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                'scales': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                'translations': [[0.0, 0.1, 0.0], [0.0, 0.6, 0.0]],
            }
            self.load_demo_n_object(demo, demo_dict)

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
                'dHat': 0.002, 'kappa': 1.0, 'elastic_type': 'NH', 'ground_barrier': 1,
                'model_paths': meshes,
                'rotations': rotations,
                'scales': scales,
                'translations': translations,
            }
            self.load_demo_n_object(demo, demo_dict)

        elif demo == 'stiff_teapots':
            meshes = ['../model/mesh/box/box.node']
            rotations = [[-90.0, 0.0, 0.0]]
            scales = [[4.0, 4.0, 4.0]]
            translations = [[0.0, 0.0, 0.0]]
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
                'dHat': 0.001, 'kappa': 1.0, 'elastic_type': 'NH', 'ground_barrier': 1,
                'model_paths': meshes,
                'rotations': rotations,
                'scales': scales,
                'translations': translations,
            }
            self.load_demo_n_object(demo, demo_dict)

        elif demo == 'stiff_two_soft_bunnies':
            demo_dict = {
                'E': 1e4, 'nu': 0.4, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.1,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'NH', 'ground_barrier': 1,
                'model_paths': ['../model/mesh/bunny_stiff/bunny_stiff.node'] * 2,
                'rotations': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                'scales': [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
                'translations': [[0.0, 0.5, 0.0], [0.0, -0.65, 0.0]],
            }
            self.load_demo_n_object(demo, demo_dict)

        # MAS Tests
        elif demo == 'eight_E_stiffness_test':
            demo_dict = {
                'E': 1e7, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 50, 'height': 0.5,
                'dHat': 0.025, 'kappa': 0.5, 'elastic_type': 'ARAP_filter', 'ground_barrier': 1,
                'model_paths': ['../model/mesh/e_2/e_2.node' for _ in range(8)],
                'rotations': [[0.0, 0.0, 0.0] for _ in range(8)],
                'scales': [[1.0, 1.0, 1.0] for _ in range(8)],
                'translations': [[1.5 * j, 1.5 * i, 0.0] for i in range(4) for j in range(2)],
                'camera_position': [2.02077697, -0.54062709, 2.59427191],
                'camera_lookat': [1.34371885, -0.79285719, 1.90291651],
                'use_mas': False,
            }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'eight_E_stiffness_mas':
            demo_dict = {
                'E': 1e7, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 200, 'height': 0.5,
                'dHat': 0.025, 'kappa': 0.5, 'elastic_type': 'SNH', 'ground_barrier': 1,
                'model_paths': ['../model/mesh/e_2/e_2.node' for _ in range(8)],
                'rotations': [[0.0, 0.0, 0.0] for _ in range(8)],
                'scales': [[1.0, 1.0, 1.0] for _ in range(8)],
                'translations': [[1.5 * j, 1.5 * i, 0.0] for i in range(4) for j in range(2)],
                'camera_position': [2.02077697, -0.54062709, 2.59427191],
                'camera_lookat': [1.34371885, -0.79285719, 1.90291651],
                'use_mas': True,
            }
            self.load_demo_n_object(demo, demo_dict)
        elif demo == 'eight_E_cubic':
            demo_dict = {
                'E': 1e4, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.5,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'SNH', 'ground_barrier': 1,
                'barrier_type': 'cubic',
                'adaptive_kappa': True,
                'cache_kappa': False,
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
            demo_dict = {
                'E': 1e4, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-4, 'iter_max': 100, 'height': 0.5,
                'dHat': 0.01, 'kappa': 1.0, 'elastic_type': 'SNH', 'ground_barrier': 1,
                'barrier_type': 'cubic',
                'adaptive_kappa': True,
                'cache_kappa': False,
                'model_paths': ['../model/mesh/e_2/e_2.node' for _ in range(8)],
                'rotations': [[0.0, 0.0, 0.0] for _ in range(8)],
                'scales': [[1.0, 1.0, 1.0] for _ in range(8)],
                'translations': [[1.5 * j, 1.5 * i, 0.0] for i in range(4) for j in range(2)],
                'camera_position': [2.02077697, -0.54062709, 2.59427191],
                'camera_lookat': [1.34371885, -0.79285719, 1.90291651],
                'use_mas': False,
            }
            self.load_demo_n_object(demo, demo_dict)

        # ========== Free-Fall Validation Tests ==========
        elif demo == 'cube_freefall':
            demo_dict = {'E': 1e5, 'nu': 0.3, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.01,
                         'epsilon': 1e-7, 'iter_max': 100, 'height': 100.0, 'elastic_type': 'ARAP_SPD',
                         'model_paths': ['../model/mesh/cube/cube.node'],
                         'rotations': [[0, 0, 0]],
                         'scales': [[1.0, 1.0, 1.0]], 'translations': [[0.0, 0.0, 0.0]],
                         }
            self.load_demo_n_object_collision_free(demo, demo_dict)

        elif demo == 'cube_freefall_10':
            demo_dict = {'E': 1e6, 'nu': 0.3, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.01,
                         'epsilon': 1e-7, 'iter_max': 100, 'height': 100.0, 'elastic_type': 'ARAP_SPD',
                         'model_paths': ['../model/mesh/cube_10/cube_10.node'],
                         'rotations': [[0, 0, 0]],
                         'scales': [[1.0, 1.0, 1.0]], 'translations': [[0.0, 0.0, 0.0]],
                         }
            self.load_demo_n_object_collision_free(demo, demo_dict)

        elif demo == 'cube_freefall_20':
            demo_dict = {'E': 1e5, 'nu': 0.3, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.01,
                         'epsilon': 1e-7, 'iter_max': 100, 'height': 100.0, 'elastic_type': 'ARAP_SPD',
                         'model_paths': ['../model/mesh/cube_20/cube_20.node'],
                         'rotations': [[0, 0, 0]],
                         'scales': [[1.0, 1.0, 1.0]], 'translations': [[0.0, 0.0, 0.0]],
                         }
            self.load_demo_n_object_collision_free(demo, demo_dict)

        elif demo == 'cube_freefall_40':
            demo_dict = {'E': 1e5, 'nu': 0.3, 'density': 1000.0, 'gravity': -9.8, 'dt': 0.01,
                         'epsilon': 1e-7, 'iter_max': 100, 'height': 100.0, 'elastic_type': 'ARAP_SPD',
                         'model_paths': ['../model/mesh/cube_40/cube_40.node'],
                         'rotations': [[0, 0, 0]],
                         'scales': [[1.0, 1.0, 1.0]], 'translations': [[0.0, 0.0, 0.0]],
                         }
            self.load_demo_n_object_collision_free(demo, demo_dict)

        elif demo == 'eight_E_freefall':
            demo_dict = {
                'E': 1e7, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-6, 'iter_max': 30, 'height': 100.0, 'elastic_type': 'ARAP_filter',
                'model_paths': ['../model/mesh/e_2/e_2.node' for _ in range(8)],
                'rotations': [[0.0, 0.0, 0.0] for _ in range(8)],
                'scales': [[1.0, 1.0, 1.0] for _ in range(8)],
                'translations': [[1.5 * j, 1.5 * i, 0.0] for i in range(4) for j in range(2)],
                'camera_position': [2.02077697, -0.54062709, 2.59427191],
                'camera_lookat': [1.34371885, -0.79285719, 1.90291651],
            }
            print('!E 1e7')
            self.load_demo_n_object_collision_free(demo, demo_dict)

        elif demo == 'eight_E_freefall_mas':
            demo_dict = {
                'E': 1e4, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-6, 'iter_max': 100, 'height': 100.0, 'elastic_type': 'ARAP_SPD',
                'model_paths': ['../model/mesh/e_2/e_2.node' for _ in range(8)],
                'rotations': [[0.0, 0.0, 0.0] for _ in range(8)],
                'scales': [[1.0, 1.0, 1.0] for _ in range(8)],
                'translations': [[1.5 * j, 1.5 * i, 0.0] for i in range(4) for j in range(2)],
                'camera_position': [2.02077697, -0.54062709, 2.59427191],
                'camera_lookat': [1.34371885, -0.79285719, 1.90291651],
                'use_mas': True,
            }
            self.load_demo_n_object_collision_free(demo, demo_dict)

        elif demo == 'eight_E_freefall_stiff':
            demo_dict = {
                'E': 1e7, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-6, 'iter_max': 200, 'height': 100.0, 'elastic_type': 'ARAP_SPD',
                'model_paths': ['../model/mesh/e_2/e_2.node' for _ in range(8)],
                'rotations': [[0.0, 0.0, 0.0] for _ in range(8)],
                'scales': [[1.0, 1.0, 1.0] for _ in range(8)],
                'translations': [[1.5 * j, 1.5 * i, 0.0] for i in range(4) for j in range(2)],
                'camera_position': [2.02077697, -0.54062709, 2.59427191],
                'camera_lookat': [1.34371885, -0.79285719, 1.90291651],
            }
            self.load_demo_n_object_collision_free(demo, demo_dict)

        elif demo == 'eight_E_freefall_stiff_mas':
            demo_dict = {
                'E': 1e7, 'nu': 0.4, 'density': 50.0, 'gravity': -9.8, 'dt': 0.01,
                'epsilon': 1e-6, 'iter_max': 200, 'height': 100.0, 'elastic_type': 'ARAP_SPD',
                'model_paths': ['../model/mesh/e_2/e_2.node' for _ in range(8)],
                'rotations': [[0.0, 0.0, 0.0] for _ in range(8)],
                'scales': [[1.0, 1.0, 1.0] for _ in range(8)],
                'translations': [[1.5 * j, 1.5 * i, 0.0] for i in range(4) for j in range(2)],
                'camera_position': [2.02077697, -0.54062709, 2.59427191],
                'camera_lookat': [1.34371885, -0.79285719, 1.90291651],
                'use_mas': True,
            }
            self.load_demo_n_object_collision_free(demo, demo_dict)

        else:
            # Try to provide helpful error message
            try:
                from demo_settings import list_demos
                available = list_demos()
                raise Exception(f"Demo '{demo}' not found. Available demos: {available[:20]}...")
            except ImportError:
                raise Exception(f"Demo '{demo}' not found")

    def set_para(self, demo_dict):
        E = demo_dict['E']
        nu = demo_dict['nu']
        self.mu, self.la = E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
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
            self.ground_barrier = int(demo_dict['ground_barrier'])
            if 'barrier_type' in demo_dict:
                self.barrier_type = demo_dict['barrier_type']
            else:
                self.barrier_type = 'log'
            if 'adaptive_kappa' in demo_dict:
                self.adaptive_kappa = demo_dict['adaptive_kappa']
            else:
                self.adaptive_kappa = False
            if 'cache_kappa' in demo_dict:
                self.cache_kappa = demo_dict['cache_kappa']
            else:
                self.cache_kappa = True

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
        if rotation != [0., 0., 0.]:
            rotation = np.asarray(rotation)
            rotation = rotation * np.pi / 180.0
            rotation_matrix = Rotation.from_rotvec(rotation).as_matrix()
            model[0] = np.dot(model[0], rotation_matrix)
        if translation != [0., 0., 0.]:
            model[0][:, 0] = model[0][:, 0] + translation[0]
            model[0][:, 1] = model[0][:, 1] + translation[1]
            model[0][:, 2] = model[0][:, 2] + translation[2]
        return model

    def load_demo_n_object_collision_free(self, demo, demo_dict):
        self.set_para(demo_dict)
        self.dict = demo_dict
        models = []
        print('add model0')
        number = len(demo_dict['scales'])
        for i in range(number):
            model_i = self.add_object(model_path=demo_dict['model_paths'][i], scale=demo_dict['scales'][i], translation=demo_dict['translations'][i], rotation=demo_dict['rotations'][i])
            models.append(model_i)
            if i == 0:
                self.ground = np.min(model_i[0][:, 1]) - demo_dict['height']

        # Apply METIS reordering for optimal MAS preconditioner performance
        reordered_dict, self.metis_result = _merge_and_reorder_models(models)
        self.mesh = Patcher.load_mesh(reordered_dict, relations=["CV"])
        self.auto_camera_from_models(models)
        print('load finish')

    def load_demo_n_object(self, demo, demo_dict):
        self.set_para(demo_dict)
        self.dict = demo_dict
        models = []
        print('add model0')
        number = len(demo_dict['scales'])
        ground_min = 100.0
        for i in range(number):
            model_i = self.add_object(model_path=demo_dict['model_paths'][i], scale=demo_dict['scales'][i], translation=demo_dict['translations'][i], rotation=demo_dict['rotations'][i])
            models.append(model_i)
            ground_min = min(ground_min, np.min(model_i[0][:, 1]))

        self.ground = ground_min - demo_dict['height']
        print('load mesh')
        self.load_mesh_and_boundarys_metis(demo, models)
        self.auto_camera_from_models(models)

    def load_demo_n_object_dirichlet(self, demo, demo_dict):
        self.set_para(demo_dict)
        self.dict = demo_dict
        models = []
        print('add model0')
        number = len(demo_dict['scales'])
        ground_min = 100.0
        for i in range(number):
            model_i = self.add_object(model_path=demo_dict['model_paths'][i], scale=demo_dict['scales'][i], translation=demo_dict['translations'][i], rotation=demo_dict['rotations'][i])
            models.append(model_i)
            ground_min = min(ground_min, np.min(model_i[0][:, 1]))

        self.ground = ground_min - demo_dict['height']
        print('load mesh')
        self.load_mesh_and_boundarys_metis(demo, models)
        self.auto_camera_from_models(models)
        self.mesh.verts.place({'is_dirichlet': ti.i32})
        dirichlet_path = demo_dict['dirichlet_path']
        dirichlet_np = np.load(dirichlet_path)
        # Apply METIS reordering to dirichlet flags if METIS was used
        if self.metis_result is not None and hasattr(self.metis_result, 'sort_index'):
            # Reorder dirichlet flags according to METIS order
            dirichlet_reordered = dirichlet_np[self.metis_result.sort_index]
            self.mesh.verts.is_dirichlet.from_numpy(dirichlet_reordered)
        else:
            self.mesh.verts.is_dirichlet.from_numpy(dirichlet_np)

    def load_mesh_and_boundarys_metis(self, demo, models):
        """
        Load mesh with METIS reordering and boundary information.

        This method:
        1. Merges models and applies METIS reordering
        2. Computes or loads boundary information
        3. Remaps boundary vertex IDs according to METIS reordering
        """
        save_path = '../demo_results/final/' + demo + '_metis/boundary/'

        # First, compute METIS reordering
        reordered_dict, self.metis_result = _merge_and_reorder_models(models)

        # Check if boundary data exists for this METIS-reordered version
        if not os.path.exists(save_path + '/boundary_points.npy'):
            self.load_and_save_boundarys_metis(demo, reordered_dict)

        # Load mesh with METIS-reordered data
        self.mesh = Patcher.load_mesh(reordered_dict, relations=["CV"])

        # Load boundary data (already in METIS-reordered vertex IDs)
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

    def load_and_save_boundarys_metis(self, demo, reordered_dict):
        """
        Compute and save boundary information using METIS-reordered mesh.

        The boundary data is saved with METIS-reordered vertex IDs so it can
        be used directly with the METIS-reordered mesh.
        """
        save_path = '../demo_results/final/' + demo + '_metis/boundary/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print('assign boundarys (METIS-reordered)... ')

        # Load mesh with METIS-reordered data for boundary computation
        self.mesh_tmp = Patcher.load_mesh(reordered_dict, relations=['FC', 'FE', 'FV', 'EV'])
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

    def load_and_save_boundarys(self, demo, models):
        save_path = '../demo_results/final/' + demo + '/boundary/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print('assign boundarys... ')
        self.mesh_tmp = Patcher.load_mesh(models, relations=['FC', 'FE', 'FV', 'EV'])
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

    def load_mesh_with_boundary_edges(self, demo, models):
        """
        Load mesh with boundary edges injected for mesh-for loop support.

        This method loads the mesh with boundary edges as element order 1,
        enabling mesh-for loops over boundary edges: `for e in mesh.edges`.

        The mesh will have:
        - mesh.verts: all vertices (with positions)
        - mesh.edges: ONLY boundary edges (not all edges)
        - mesh.cells: all tetrahedral cells
        """
        save_path = '../demo_results/final/' + demo + '_metis/boundary/'

        # First, compute METIS reordering
        reordered_dict, self.metis_result = _merge_and_reorder_models(models)

        # Check if boundary data exists
        if not os.path.exists(save_path + '/boundary_points.npy'):
            self.load_and_save_boundarys_metis(demo, reordered_dict)

        # Load boundary data
        boundary_points_np = np.load(save_path + '/boundary_points.npy')
        boundary_edges_np = np.load(save_path + '/boundary_edges.npy')
        boundary_triangles_np = np.load(save_path + '/boundary_triangles.npy')

        n_boundary_points = boundary_points_np.shape[0]
        n_boundary_edges = boundary_edges_np.shape[0]
        n_boundary_triangles = boundary_triangles_np.shape[0]
        print(f'[load_mesh_with_boundary_edges] Boundary elements:')
        print(f'  Points: {n_boundary_points}, Edges: {n_boundary_edges}, Triangles: {n_boundary_triangles}')

        # Inject boundary edges into mesh dict
        # Key 1 = edges, this tells patcher to use these edges instead of generating all edges
        mesh_dict_with_edges = {
            0: reordered_dict[0],  # vertices
            1: boundary_edges_np.astype(np.int32),  # boundary edges only
            3: reordered_dict[3],  # cells
        }

        # Load mesh with CV and EV relations
        # Since we injected boundary edges as element 1, mesh.edges will only contain boundary edges
        print('[load_mesh_with_boundary_edges] Loading mesh with injected boundary edges...')
        self.mesh = Patcher.load_mesh(mesh_dict_with_edges, relations=["CV", "EV"])

        print(f'[load_mesh_with_boundary_edges] Mesh loaded:')
        print(f'  Vertices: {len(self.mesh.verts)}')
        print(f'  Edges (boundary): {len(self.mesh.edges)}')
        print(f'  Cells: {len(self.mesh.cells)}')

        # Verify edge count matches boundary edges
        if len(self.mesh.edges) != n_boundary_edges:
            print(f'  WARNING: Edge count mismatch! Expected {n_boundary_edges}, got {len(self.mesh.edges)}')

        # Also store boundary data in old format for compatibility
        self.boundary_points = ti.field(ti.i32)
        self.boundary_edges = ti.field(ti.i32)
        self.boundary_triangles = ti.field(ti.i32)
        ti.root.dense(ti.i, n_boundary_points).place(self.boundary_points)
        ti.root.dense(ti.ij, (n_boundary_edges, 2)).place(self.boundary_edges)
        ti.root.dense(ti.ij, (n_boundary_triangles, 3)).place(self.boundary_triangles)
        self.boundary_points.from_numpy(boundary_points_np)
        self.boundary_edges.from_numpy(boundary_edges_np)
        self.boundary_triangles.from_numpy(boundary_triangles_np)
        print('[load_mesh_with_boundary_edges] Done!')

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
            self.triangles[id, 0] = f.verts[0].id
            self.triangles[id, 1] = f.verts[1].id
            self.triangles[id, 2] = f.verts[2].id
        for e in self.mesh_tmp.edges:
            id = e.id
            self.edges[id, 0] = e.verts[0].id
            self.edges[id, 1] = e.verts[1].id

    def auto_camera_from_models(self, models, fov_degrees=45.0, padding=1.5):
        """
        Compute camera position and lookat from loaded models.
        Only computes if camera_position/camera_lookat not already set in demo_dict.
        """
        if 'camera_position' in self.dict and 'camera_lookat' in self.dict:
            self.camera_position = self.dict['camera_position']
            self.camera_lookat = self.dict['camera_lookat']
        else:
            all_vertices = np.vstack([model[0] for model in models])
            self.camera_position, self.camera_lookat = compute_auto_camera(
                all_vertices, fov_degrees, padding
            )
