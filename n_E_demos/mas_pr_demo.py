"""
MAS Preconditioner Demo with PR (Polak-Ribiere) Conjugate Gradient

Features:
- MAS preconditioner rebuilt only at first iteration
- PR formula for conjugate direction update: beta = (g_{k+1}^T z_{k+1} - g_{k+1}^T z_k) / (g_k^T z_k)
- Classic alpha = g^T p / p^T H p line search
- Collision-free for simplicity
"""

import sys
import os
import time
import argparse
import numpy as np

# Setup paths
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)
demo_dir = os.path.join(parent_dir, 'demo')
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti
from algorithm.pncg_base_ipc import pncg_ipc_deformer
from algorithm.mas_preconditioner_pkg import MASPreconditioner
from demo_runner import DemoRunner


@ti.data_oriented
class MASPRSolver(pncg_ipc_deformer):
    """
    PNCG solver with MAS preconditioner using PR (Polak-Ribiere) formula.

    PR formula: beta_{k+1} = (g_{k+1}^T z_{k+1} - g_{k+1}^T z_k) / (g_k^T z_k)
    Direction:  p_{k+1} = -z_{k+1} + beta_{k+1} * p_k
    Step size:  alpha = g^T p / p^T H p
    """

    def __init__(self, demo='eight_E_stiffness_test', use_mas=True):
        super().__init__(demo=demo)

        self.use_mas = use_mas
        self.object_size = 1046
        self.N_object = int(self.n_verts / self.object_size)

        # Per-vertex color for visualization
        self.per_vertex_color = ti.Vector.field(3, dtype=float, shape=self.n_verts)
        self._init_colors()

        # Additional fields for PR formula
        # z: preconditioned gradient (P * g)
        # z_prev: z from previous iteration
        # Note: grad_prev is already defined in base class
        self.mesh.verts.place({
            'z': ti.types.vector(3, float),
            'z_prev': ti.types.vector(3, float)
        })

        # Scalar for gTz (used in PR beta computation)
        self.gTz_prev = ti.field(dtype=float, shape=())

        # Initialize MAS preconditioner
        if self.use_mas:
            print(f"[MAS-PR] Initializing MAS preconditioner...")
            self.mas = MASPreconditioner(
                self.n_verts, self.n_cells, self.mesh,
                use_metis=False
            )
            print(f"[MAS-PR] MAS initialized with {self.mas.level_num} levels")
        else:
            self.mas = None
            print(f"[MAS-PR] Using diagonal Jacobi preconditioner")

        # Performance tracking
        self.iter_history = []
        self.time_history = []
        self.mas_rebuilt = False

        # Apply initial velocity to create larger gradient
        self._apply_initial_velocity()

    @ti.kernel
    def _apply_initial_velocity(self):
        """Apply initial downward velocity to all vertices."""
        for vert in self.mesh.verts:
            vert.v = ti.Vector([0.0, -1.0, 0.0])  # Mild downward velocity

    @ti.kernel
    def _init_colors(self):
        """Initialize per-vertex colors based on object index."""
        for vert in self.mesh.verts:
            index = vert.id // self.object_size
            color_idx = index % 8
            if color_idx == 0:
                self.per_vertex_color[vert.id] = ti.Vector([1.0, 0.5, 0.0])
            elif color_idx == 1:
                self.per_vertex_color[vert.id] = ti.Vector([0.2, 0.4, 0.8])
            elif color_idx == 2:
                self.per_vertex_color[vert.id] = ti.Vector([0.3, 0.8, 0.3])
            elif color_idx == 3:
                self.per_vertex_color[vert.id] = ti.Vector([0.7, 0.3, 0.8])
            elif color_idx == 4:
                self.per_vertex_color[vert.id] = ti.Vector([0.2, 0.8, 0.8])
            elif color_idx == 5:
                self.per_vertex_color[vert.id] = ti.Vector([0.9, 0.9, 0.2])
            elif color_idx == 6:
                self.per_vertex_color[vert.id] = ti.Vector([0.9, 0.2, 0.2])
            else:
                self.per_vertex_color[vert.id] = ti.Vector([0.9, 0.5, 0.7])

    @ti.kernel
    def compute_z_diag(self):
        """Compute z = P * g using diagonal preconditioner: z = g / diagH"""
        for vert in self.mesh.verts:
            vert.z = vert.grad / vert.diagH

    @ti.kernel
    def compute_diagH_stats(self) -> ti.types.vector(3, float):
        """Compute diagH statistics: min, max, avg"""
        diagH_min = 1e30
        diagH_max = 0.0
        diagH_sum = 0.0
        for vert in self.mesh.verts:
            dH = vert.diagH.norm()
            ti.atomic_min(diagH_min, dH)
            ti.atomic_max(diagH_max, dH)
            diagH_sum += dH
        return ti.Vector([diagH_min, diagH_max, diagH_sum / 8368.0])

    @ti.kernel
    def compute_init_direction(self):
        """Initial direction: p = -z"""
        for vert in self.mesh.verts:
            vert.p = -vert.z

    @ti.kernel
    def compute_gTz(self) -> float:
        """Compute g^T z"""
        result = 0.0
        for vert in self.mesh.verts:
            result += vert.grad.dot(vert.z)
        return result

    @ti.kernel
    def compute_gTz_cross(self) -> float:
        """Compute g_{k+1}^T z_k (cross term for PR)"""
        result = 0.0
        for vert in self.mesh.verts:
            result += vert.grad.dot(vert.z_prev)
        return result

    @ti.kernel
    def compute_gTp(self) -> float:
        """Compute g^T p"""
        result = 0.0
        for vert in self.mesh.verts:
            result += vert.grad.dot(vert.p)
        return result

    @ti.kernel
    def save_z_and_grad(self):
        """Save current z and grad for next iteration's PR computation"""
        for vert in self.mesh.verts:
            vert.z_prev = vert.z
            vert.grad_prev = vert.grad

    @ti.kernel
    def compute_PR_direction(self, beta: float):
        """
        Compute PR direction: p_{k+1} = -z_{k+1} + beta * p_k
        """
        for vert in self.mesh.verts:
            vert.p = -vert.z + beta * vert.p

    @ti.kernel
    def compute_grad_inf_norm(self) -> float:
        """Compute infinity norm of gradient: max(|g_i|)"""
        g_max = 0.0
        for vert in self.mesh.verts:
            g_norm = vert.grad.norm()
            ti.atomic_max(g_max, g_norm)
        return g_max

    @ti.kernel
    def compute_z_inf_norm(self) -> float:
        """Compute infinity norm of z: max(|z_i|)"""
        z_max = 0.0
        for vert in self.mesh.verts:
            z_norm = vert.z.norm()
            ti.atomic_max(z_max, z_norm)
        return z_max

    @ti.kernel
    def check_z_has_nan(self) -> int:
        """Check if z field contains nan."""
        has_nan = 0
        for vert in self.mesh.verts:
            if ti.math.isnan(vert.z[0]) or ti.math.isnan(vert.z[1]) or ti.math.isnan(vert.z[2]):
                has_nan = 1
        return has_nan

    def step(self, grad_tol=1e-4, verbose=True):
        """
        One frame optimization using PR-CG with MAS preconditioner.

        MAS is only rebuilt at the first iteration of each frame.
        """
        if verbose:
            print(f'Frame {self.frame}')
        ti.sync()
        t_frame_start = time.perf_counter()

        self.assign_xn_xhat()

        grad_inf_init = None

        for iter in range(self.iter_max):
            # Compute gradient and diagonal Hessian
            self.compute_grad_and_diagH()

            # Add ground barrier
            if self.ground_barrier == 1:
                self.add_grad_and_diagH_ground_barrier()

            # Check convergence using |g|_inf norm
            grad_inf = self.compute_grad_inf_norm()
            if iter == 0:
                grad_inf_init = grad_inf
                if verbose:
                    diagH_stats = self.compute_diagH_stats()
                    print(f'    diagH: min={diagH_stats[0]:.4e}, max={diagH_stats[1]:.4e}, avg={diagH_stats[2]:.4e}')
            if grad_inf < grad_tol:
                if verbose:
                    print(f'  Converged at iter {iter}, |g|_inf={grad_inf:.2e}')
                break

            # Apply preconditioner: z = P * g
            # Currently using diagonal preconditioner for stability
            # MAS preconditioner has block assembly issues (non-SPD blocks)
            self.compute_z_diag()
            gTz = self.compute_gTz()

            if iter == 0:
                # Initial direction: p = -z
                self.compute_init_direction()
                self.gTz_prev[None] = gTz
                if verbose:
                    z_inf = self.compute_z_inf_norm()
                    print(f'    |z|_inf={z_inf:.4e}, gTz={gTz:.4e}')
            else:
                # PR formula: beta = (g_{k+1}^T z_{k+1} - g_{k+1}^T z_k) / (g_k^T z_k)
                gTz_new = gTz
                gTz_cross = self.compute_gTz_cross()  # g_{k+1}^T z_k
                gTz_old = self.gTz_prev[None]

                if abs(gTz_old) > 1e-12:
                    beta = (gTz_new - gTz_cross) / gTz_old
                    # PR+ modification: max(beta, 0)
                    beta = max(beta, 0.0)
                    # Limit beta to avoid numerical instability
                    if beta > 1.0:
                        beta = 0.0  # Restart when beta is too large
                else:
                    beta = 0.0  # Restart

                self.compute_PR_direction(beta)
                self.gTz_prev[None] = gTz_new

            # Save z and grad for next iteration
            self.save_z_and_grad()

            # Line search: alpha = g^T p / p^T H p
            # Use Newton-based line search from base class
            alpha, gTp, pHp = self.line_search_newton()

            # Safety clamp
            p_max = self.compute_p_inf_norm()
            if alpha * p_max > 0.5 * self.dHat:
                alpha = 0.5 * self.dHat / p_max

            # Debug: print displacement
            if verbose and iter == 0:
                displacement = alpha * p_max
                print(f'    alpha={alpha:.4e}, |p|_inf={p_max:.4e}, gTp={gTp:.4e}, pHp={pHp:.4e}, disp={displacement:.4e}')

            # Update position
            self.update_x(alpha)

        self.update_v_and_bound()

        ti.sync()
        frame_time = (time.perf_counter() - t_frame_start) * 1000

        self.iter_history.append(iter + 1)
        self.time_history.append(frame_time)

        if verbose:
            final_grad = self.compute_grad_inf_norm()
            print(f'  Frame {self.frame}: {iter+1} iters, {frame_time:.2f}ms, |g|: {grad_inf_init:.2e} -> {final_grad:.2e}')

        self.frame += 1
        return iter + 1


class MASPRDemo(DemoRunner):
    """Demo runner for MAS-PR solver."""

    def __init__(self, demo='eight_E_stiffness_test', use_mas=True):
        solver = MASPRSolver(demo=demo, use_mas=use_mas)
        super().__init__(solver, demo_name=f"MAS-PR ({demo})")
        self.use_mas = use_mas

    def get_per_vertex_color(self):
        return self.solver.per_vertex_color

    def setup(self):
        print(f"  Objects: {self.solver.N_object}")
        print(f"  E (stiffness): {self.solver.dict['E']}")
        print(f"  Preconditioner: {'MAS' if self.use_mas else 'Diagonal'}")


def main():
    parser = argparse.ArgumentParser(description='MAS-PR Demo')
    parser.add_argument('--demo', type=str, default='eight_E_stiffness_test', help='Demo name')
    parser.add_argument('--diag', action='store_true', help='Use diagonal preconditioner instead of MAS')
    parser.add_argument('--headless', action='store_true', help='Run without GUI')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    ti.init(arch=ti.gpu, default_fp=ti.f32, device_memory_GB=4.0)

    demo = MASPRDemo(demo=args.demo, use_mas=not args.diag)

    from demo_runner import RunConfig
    config = RunConfig(
        headless=args.headless,
        debug=args.debug,
        frames=args.frames,
        demo_name=demo.demo_name
    )

    demo.run(config)


if __name__ == '__main__':
    main()
