"""
Example demo configurations.

These serve as reference implementations and can be used directly or as
templates for creating new demo configurations.
"""

from config.base import (
    SimulationConfig,
    MaterialConfig,
    SolverConfig,
    SceneConfig,
    MeshInstance,
    IPCConfig,
)
from config.registry import register_demo


# =============================================================================
# Collision-Free Demos
# =============================================================================

@register_demo("cube")
def cube_config():
    """Basic cube demo for collision-free deformation."""
    return SimulationConfig(
        name="cube",
        material=MaterialConfig(
            E=1e4,
            nu=0.4,
            density=500.0,
            elastic_type="ARAP_filter"
        ),
        solver=SolverConfig(
            epsilon=1e-10,
            iter_max=50,
            dt=0.04
        ),
        scene=SceneConfig(
            meshes=[MeshInstance(path="../model/mesh/cube/cube.node")],
            ground_height=4.0,
            gravity=-9.8,
            camera_position=[0.16850185, -1.69999744, 5.03710925],
            camera_lookat=[0.14355508, -1.50024417, 4.05758065]
        )
    )


@register_demo("cube_10")
def cube_10_config():
    """Cube with 10x10x10 resolution."""
    return SimulationConfig(
        name="cube_10",
        material=MaterialConfig(
            E=1e4,
            nu=0.4,
            density=500.0,
            elastic_type="ARAP_filter"
        ),
        solver=SolverConfig(
            epsilon=1e-7,
            iter_max=100,
            dt=0.04
        ),
        scene=SceneConfig(
            meshes=[MeshInstance(path="../model/mesh/cube_10/cube_10.node")],
            ground_height=5.0,
            gravity=-9.8,
            camera_position=[0.44853318, 0.50973239, 0.34616475],
            camera_lookat=[0.4503797, 0.53796244, -0.653435]
        )
    )


@register_demo("cube_20")
def cube_20_config():
    """Cube with 20x20x20 resolution."""
    return SimulationConfig(
        name="cube_20",
        material=MaterialConfig(
            E=1e4,
            nu=0.4,
            density=500.0,
            elastic_type="ARAP_filter"
        ),
        solver=SolverConfig(
            epsilon=1e-5,
            iter_max=100,
            dt=0.01
        ),
        scene=SceneConfig(
            meshes=[MeshInstance(
                path="../model/mesh/cube_20/cube_20.node",
                rotation=[90, 0, 0]
            )],
            ground_height=1.0,
            gravity=-9.8,
            camera_position=[0.44853318, 0.50973239, 0.34616475],
            camera_lookat=[0.4503797, 0.53796244, -0.653435]
        )
    )


@register_demo("cube_40")
def cube_40_config():
    """Cube with 40x40x40 resolution using SNH material."""
    return SimulationConfig(
        name="cube_40",
        material=MaterialConfig(
            E=1e4,
            nu=0.4,
            density=500.0,
            elastic_type="SNH"
        ),
        solver=SolverConfig(
            epsilon=1e-5,
            iter_max=50,
            dt=0.04
        ),
        scene=SceneConfig(
            meshes=[MeshInstance(
                path="../model/mesh/cube_40/cube_40.node",
                rotation=[90, 0, 0]
            )],
            ground_height=4.0,
            gravity=-9.8,
            camera_position=[0.4426824, 1.04414034, 1.44265061],
            camera_lookat=[0.45726201, 1.19416167, 0.45407536]
        )
    )


# =============================================================================
# IPC Contact Demos
# =============================================================================

@register_demo("eight_E_drop_demo_contact")
def eight_e_drop_config():
    """Eight E-shaped objects dropping with IPC contact."""
    # Create 8 mesh instances in a 4x2 grid
    meshes = [
        MeshInstance(
            path="../model/mesh/e_2/e_2.node",
            translation=[1.5 * j, 1.5 * i, 0.0]
        )
        for i in range(4) for j in range(2)
    ]

    return SimulationConfig(
        name="eight_E_drop_demo_contact",
        material=MaterialConfig(
            E=1e4,
            nu=0.4,
            density=50.0,
            elastic_type="NH"
        ),
        solver=SolverConfig(
            epsilon=1e-4,
            iter_max=50,
            dt=0.01
        ),
        scene=SceneConfig(
            meshes=meshes,
            ground_height=0.5,
            gravity=-9.8,
            camera_position=[2.02077697, -0.54062709, 2.59427191],
            camera_lookat=[1.34371885, -0.79285719, 1.90291651]
        ),
        ipc=IPCConfig(
            enabled=True,
            dHat=0.025,
            kappa=0.5,
            adj=0,
            ground_barrier=True
        )
    )


# =============================================================================
# Dirichlet Boundary Condition Demos
# =============================================================================

@register_demo("twist_rods")
def twist_rods_config():
    """Four rods with twisting Dirichlet boundary conditions."""
    meshes = [
        MeshInstance(
            path="../model/mesh/rod300x33/rod300x33.node",
            translation=trans
        )
        for trans in [
            [0.0, -0.1, -0.1],
            [0.0, -0.1, 0.1],
            [0.0, 0.1, -0.1],
            [0.0, 0.1, 0.1]
        ]
    ]

    return SimulationConfig(
        name="twist_rods",
        material=MaterialConfig(
            E=1e4,
            nu=0.4,
            density=1000.0,
            elastic_type="FCR_filter"
        ),
        solver=SolverConfig(
            epsilon=1e-3,
            iter_max=150,
            dt=0.04
        ),
        scene=SceneConfig(
            meshes=meshes,
            ground_height=0.0,
            gravity=0.0,
            camera_position=[-0.09454866, -0.05231105, -0.56085817],
            camera_lookat=[-0.01724642, 0.03024886, 0.43272535]
        ),
        ipc=IPCConfig(
            enabled=True,
            dHat=0.001,
            kappa=0.1,
            adj=0,
            ground_barrier=False
        ),
        dirichlet_path="../model/mesh/rod300x33/is_dirichlet.npy"
    )
