"""
Eight E Free-Fall Demo - Validates MAS solver with 8 E-shaped objects in free fall.

This demo validates the MAS preconditioner correctness by comparing simulation results
with Newton's laws ground truth in a collision-free free-fall scenario.

Ground Truth (Newton's Laws):
- Position: y(t) = y0 + v0*t + 0.5*g*t^2
- Velocity: v(t) = v0 + g*t

The 8 E-shaped objects fall freely under gravity with an initial downward velocity.
No collision detection, no ground barrier - pure elastic + inertia.

Usage:
    python eight_E_freefall_demo.py                          # Test with diagonal preconditioner
    python eight_E_freefall_demo.py --use-mas                # Test with MAS preconditioner
    python eight_E_freefall_demo.py --frames 10              # Test with 10 frames
    python eight_E_freefall_demo.py --compare                # Compare MAS vs diagonal
"""

import sys
import os
import time
import argparse
import numpy as np

# Setup paths - find project root (PNCG_IPC) and set up properly
current_file_path = os.path.abspath(__file__)
n_E_demos_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(n_E_demos_dir)
demo_dir = os.path.join(project_root, 'demo')

# Add project root and demo to path
sys.path.insert(0, project_root)
sys.path.insert(0, demo_dir)
os.chdir(demo_dir)

import taichi as ti
from math_utils.elastic_util import *
from util.model_loading import model_loading
from algorithm.mas_preconditioner_pkg import MASPreconditioner


@ti.data_oriented
class EightEFreeFallValidator:
    """
    Free-fall validator for 8 E-shaped objects with MAS preconditioner.

    Uses a collision-free setup where 8 E objects fall under gravity.
    Compares centroid motion with Newton's law prediction.
    """

    def __init__(self, demo='eight_E_freefall', inversion_method='oneway_gj', use_mas=False):
        """
        Args:
            demo: Demo configuration name
            inversion_method: Block inversion method for MAS
            use_mas: If True, use MAS preconditioner; if False, use diagonal preconditioner
        """
        self.use_mas = use_mas

        # Load model
        model = model_loading(demo=demo)
        self.demo = demo
        self.dict = model.dict
        self.mu, self.la = model.mu, model.la
        self.density = model.density
        self.dt = model.dt
        self.gravity = model.gravity
        self.ground = model.ground
        self.mesh = model.mesh
        self.epsilon = model.epsilon
        self.iter_max = model.iter_max
        self.camera_position = model.camera_position
        self.camera_lookat = model.camera_lookat
        self.frame = 0
        self.inversion_method = inversion_method

        # Place vertex fields
        self.mesh.verts.place({
            'x': ti.types.vector(3, float),
            'v': ti.types.vector(3, float),
            'm': float,
            'x_n': ti.types.vector(3, float),
            'x_hat': ti.types.vector(3, float),
            'x_prev': ti.types.vector(3, float),
            'x_init': ti.types.vector(3, float),
            'grad': ti.types.vector(3, float),
            'grad_prev': ti.types.vector(3, float),
            'p': ti.types.vector(3, float),
            'diagH': ti.types.vector(3, float),
            'z': ti.types.vector(3, float),
        })

        # Place cell fields
        self.mesh.cells.place({'B': ti.math.mat3, 'W': float})

        # Initialize positions from mesh
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.mesh.verts.x_init.copy_from(self.mesh.verts.x)
        self.mesh.verts.x_prev.copy_from(self.mesh.verts.x)
        self.mesh.verts.v.fill([0.0, 0.0, 0.0])

        self.n_verts = len(self.mesh.verts)
        self.n_cells = len(self.mesh.cells)
        print(f"Mesh: {self.n_verts} vertices, {self.n_cells} cells")

        # Calculate number of objects (each E object has 1046 vertices)
        self.object_size = 1046
        self.n_objects = self.n_verts // self.object_size
        print(f"Number of E objects: {self.n_objects}")

        # Precompute (mass, B, W)
        self.precompute()

        # Assign elastic type
        self.assign_elastic_type(model.elastic_type)

        # Initialize MAS preconditioner (optional)
        if self.use_mas:
            print(f"[Eight E FreeFall] Initializing MAS preconditioner...")
            self.mas = MASPreconditioner(
                self.n_verts, self.n_cells, self.mesh,
                use_metis=False
            )
            print(f"[Eight E FreeFall] MAS initialized with {self.mas.level_num} levels")
        else:
            print(f"[Eight E FreeFall] Using diagonal preconditioner (no MAS)")
            self.mas = None

        # Ground truth tracking
        self.initial_centroid = np.zeros(3)
        self.initial_velocity = np.zeros(3)
        self.time_elapsed = 0.0

        # Results storage
        self.frame_results = []

    def assign_elastic_type(self, elastic):
        """Set elastic type functions.

        MAS preconditioner uses integer elastic_type:
        0=ARAP, 1=SNH, 2=FCR, 3=ARAP_SPD, 4=NH_SPD, 5=STVK_SPD
        """
        if elastic == 'ARAP':
            self.compute_dPsidx = compute_dPsidx_ARAP
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_ARAP
            self.elastic_type = 0
        elif elastic == 'SNH':
            self.compute_dPsidx = compute_dPsidx_SNH
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_SNH
            self.elastic_type = 1
        elif elastic == 'ARAP_filter':
            self.compute_dPsidx = compute_dPsidx_ARAP
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_ARAP_filter
            self.elastic_type = 0  # MAS uses ARAP for ARAP_filter
        elif elastic == 'FCR':
            self.compute_dPsidx = compute_dPsidx_FCR
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_FCR
            self.elastic_type = 2
        elif elastic == 'FCR_filter':
            self.compute_dPsidx = compute_dPsidx_FCR
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_FCR_filter
            self.elastic_type = 2  # MAS uses FCR for FCR_filter
        elif elastic == 'NH':
            self.compute_dPsidx = compute_dPsidx_NH
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_NH
            self.elastic_type = 1  # Map NH to SNH for MAS assembly
        # SPD-projected Hessian materials (eigenanalysis-based)
        elif elastic == 'ARAP_SPD':
            self.compute_dPsidx = compute_dPsidx_ARAP_SPD
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_ARAP_SPD
            self.elastic_type = 3
        elif elastic == 'NH_SPD':
            self.compute_dPsidx = compute_dPsidx_NH_SPD
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_NH_SPD
            self.elastic_type = 4
        elif elastic == 'STVK_SPD':
            self.compute_dPsidx = compute_dPsidx_STVK_SPD
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_STVK_SPD
            self.elastic_type = 5
        else:
            print(f'Warning: Unknown elastic type {elastic}, using ARAP_SPD')
            self.compute_dPsidx = compute_dPsidx_ARAP_SPD
            self.compute_diag_d2Psidx2 = compute_diag_d2Psidx2_ARAP_SPD
            self.elastic_type = 3
        self.elastic_type_str = elastic

    @ti.kernel
    def precompute(self):
        """Precompute mass, B matrix, and cell volumes."""
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            c.B = Ds.inverse()
            c.W = ti.abs(Ds.determinant()) / 6.0
            for i in ti.static(range(4)):
                c.verts[i].m += self.density * c.W / 4.0

    def compute_centroid(self):
        """Compute mass-weighted centroid of the mesh."""
        x_np = self.mesh.verts.x.to_numpy()
        m_np = self.mesh.verts.m.to_numpy()
        total_mass = np.sum(m_np)
        centroid = np.sum(x_np * m_np[:, np.newaxis], axis=0) / total_mass
        return centroid, total_mass

    def compute_velocity_centroid(self):
        """Compute mass-weighted velocity of centroid."""
        v_np = self.mesh.verts.v.to_numpy()
        m_np = self.mesh.verts.m.to_numpy()
        total_mass = np.sum(m_np)
        v_centroid = np.sum(v_np * m_np[:, np.newaxis], axis=0) / total_mass
        return v_centroid

    def newton_ground_truth(self, t):
        """
        Compute ground truth position and velocity using Newton's laws.

        y(t) = y0 + v0*t + 0.5*g*t^2
        v(t) = v0 + g*t
        """
        g = np.array([0.0, self.gravity, 0.0])
        pos = self.initial_centroid + self.initial_velocity * t + 0.5 * g * t * t
        vel = self.initial_velocity + g * t
        return pos, vel

    def set_initial_velocity(self, vy=-1.0):
        """Set initial downward velocity for all vertices."""
        self.init_v(vy)
        self.initial_velocity = np.array([0.0, vy, 0.0])
        self.initial_centroid, _ = self.compute_centroid()
        print(f"[Eight E FreeFall] Initial centroid: {self.initial_centroid}")
        print(f"[Eight E FreeFall] Initial velocity: {self.initial_velocity}")

    @ti.kernel
    def init_v(self, vy: float):
        """Initialize velocity for all vertices."""
        for vert in self.mesh.verts:
            vert.v[1] = vy

    @ti.kernel
    def assign_xn_xhat(self):
        """Assign x_n and x_hat for implicit time integration."""
        for vert in self.mesh.verts:
            vert.x_n = vert.x
            vert.x_hat = vert.x + self.dt * vert.v
            vert.x_hat[1] += self.gravity * self.dt * self.dt

    @ti.kernel
    def compute_grad_and_diagH(self):
        """Compute gradient and diagonal Hessian for elastic + inertia."""
        # Initialize with inertia term
        ti.mesh_local(self.mesh.verts.grad)
        for vert in self.mesh.verts:
            m = vert.m
            vert.grad_prev = vert.grad
            vert.grad = m * (vert.x - vert.x_hat)
            vert.diagH = m * ti.Vector.one(float, 3)

        # Add elastic term
        for c in self.mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[0].x for i in ti.static(range(1, 4))])
            B = c.B
            F = Ds @ B
            para = c.W * self.dt ** 2

            # Compute elastic gradient (returns 12x1 vector)
            dPsidx = para * self.compute_dPsidx(F, B, self.mu, self.la)
            # Compute diagonal Hessian (returns 12x1 vector)
            diagH_d2Psidx2 = para * self.compute_diag_d2Psidx2(F, B, self.mu, self.la)

            # Add to vertex gradients and diagonal Hessian
            for i in range(4):
                c.verts[i].grad += ti.Vector([dPsidx[3 * i], dPsidx[3 * i + 1], dPsidx[3 * i + 2]], float)
                tmp = ti.Vector([diagH_d2Psidx2[3 * i], diagH_d2Psidx2[3 * i + 1], diagH_d2Psidx2[3 * i + 2]])
                tmp = ti.max(tmp, 0.0)
                c.verts[i].diagH += tmp

    @ti.kernel
    def compute_init_p(self):
        """Compute initial search direction: p = -z (preconditioned gradient)."""
        for vert in self.mesh.verts:
            vert.p = -vert.z

    @ti.kernel
    def compute_DK_direction(self):
        """Compute DK direction (simplified: steepest descent with MAS)."""
        for vert in self.mesh.verts:
            vert.p = -vert.z

    @ti.kernel
    def line_search_newton(self) -> ti.types.vector(3, float):
        """Newton line search: compute alpha = g^T p / p^T H p."""
        gTp = 0.0
        pHp = 0.0
        for vert in self.mesh.verts:
            gTp += vert.grad.dot(vert.p)
            pHp += vert.p.dot(vert.diagH * vert.p)

        alpha = 0.0
        if pHp > 1e-10:
            alpha = -gTp / pHp
        alpha = ti.max(0.0, ti.min(alpha, 1.0))

        return ti.Vector([alpha, gTp, pHp])

    @ti.kernel
    def update_x(self, alpha: float):
        """Update position: x = x + alpha * p."""
        for vert in self.mesh.verts:
            vert.x += alpha * vert.p

    @ti.kernel
    def update_v(self):
        """Update velocity: v = (x - x_n) / dt."""
        for vert in self.mesh.verts:
            vert.v = (vert.x - vert.x_n) / self.dt

    @ti.kernel
    def compute_grad_inf_norm(self) -> float:
        """Compute infinity norm of gradient."""
        g_max = 0.0
        for vert in self.mesh.verts:
            g_norm = vert.grad.norm()
            ti.atomic_max(g_max, g_norm)
        return g_max

    @ti.kernel
    def check_z_has_nan(self) -> int:
        """Check if z field contains nan."""
        has_nan = 0
        for vert in self.mesh.verts:
            if ti.math.isnan(vert.z[0]) or ti.math.isnan(vert.z[1]) or ti.math.isnan(vert.z[2]):
                has_nan = 1
        return has_nan

    def step(self, verbose=False, grad_tol=1e-5):
        """
        One time step of implicit integration with MAS preconditioner.
        """
        t_start = time.perf_counter()

        self.assign_xn_xhat()

        converged = False
        for iter in range(self.iter_max):
            # Compute gradient and diagonal Hessian
            self.compute_grad_and_diagH()

            # Check convergence
            grad_inf = self.compute_grad_inf_norm()
            if verbose and iter % 10 == 0:
                print(f'    iter {iter}: |g|_inf={grad_inf:.2e}')
            if grad_inf < grad_tol:
                if verbose:
                    print(f'  Converged at iter {iter}, |g|_inf={grad_inf:.2e}')
                converged = True
                break

            if self.use_mas and self.mas is not None:
                # Rebuild MAS at iter 0
                if iter == 0:
                    if not self.mas.hierarchy_built:
                        self.mas.build_hierarchy()
                    self.mas.assemble_block_matrices(self, use_full_hessian=True)

                    # Map old method names to new API
                    method_map = {
                        'gauss_jordan': 'gauss_jordan',
                        'oneway_gj': 'oneway_gj',
                        'cholesky': 'cholesky',
                        'incomplete': 'ic',
                    }
                    method = method_map.get(self.inversion_method, 'ic')

                    self.mas.invert_block_matrices(method=method)

                # Apply MAS preconditioner: z = P * grad
                self.mas.apply()

                # Check for NaN
                if self.check_z_has_nan():
                    if verbose or iter == 0:
                        print(f'  [Warning] MAS produced NaN at iter {iter}, using diagonal fallback')
                    # Fallback to diagonal preconditioner
                    self.fallback_diagonal_preconditioner()
            else:
                # Use diagonal preconditioner
                self.fallback_diagonal_preconditioner()

            # Compute search direction
            if iter == 0:
                self.compute_init_p()
            else:
                self.compute_DK_direction()

            # Line search
            result = self.line_search_newton()
            alpha, gTp, pHp = result[0], result[1], result[2]

            if verbose and iter == 0:
                print(f'    alpha={alpha:.4f}, gTp={gTp:.2e}, pHp={pHp:.2e}')

            # Update position
            self.update_x(alpha)

        if not converged and verbose:
            print(f'  Did not converge after {self.iter_max} iterations, |g|_inf={grad_inf:.2e}')

        # Update velocity
        self.update_v()

        # Update time
        self.time_elapsed += self.dt
        self.frame += 1

        t_elapsed = (time.perf_counter() - t_start) * 1000

        return iter + 1, t_elapsed

    @ti.kernel
    def fallback_diagonal_preconditioner(self):
        """Fallback to diagonal preconditioner when MAS fails."""
        for vert in self.mesh.verts:
            for i in ti.static(range(3)):
                if vert.diagH[i] > 1e-10:
                    vert.z[i] = vert.grad[i] / vert.diagH[i]
                else:
                    vert.z[i] = vert.grad[i]

    def validate_frame(self, verbose=True):
        """
        Validate current frame against Newton's ground truth.

        Returns:
            dict with simulation and ground truth results
        """
        # Get simulation results
        sim_centroid, total_mass = self.compute_centroid()
        sim_velocity = self.compute_velocity_centroid()

        # Get ground truth
        gt_pos, gt_vel = self.newton_ground_truth(self.time_elapsed)

        # Compute errors
        pos_error = np.linalg.norm(sim_centroid - gt_pos)
        vel_error = np.linalg.norm(sim_velocity - gt_vel)

        # Relative errors (use magnitude of ground truth as reference)
        displacement = np.linalg.norm(gt_pos - self.initial_centroid)
        pos_rel_error = pos_error / (displacement + 1e-10) if displacement > 1e-10 else pos_error
        vel_rel_error = vel_error / (np.linalg.norm(gt_vel) + 1e-10)

        result = {
            'frame': self.frame,
            'time': self.time_elapsed,
            'sim_centroid': sim_centroid.copy(),
            'gt_centroid': gt_pos.copy(),
            'sim_velocity': sim_velocity.copy(),
            'gt_velocity': gt_vel.copy(),
            'pos_error': pos_error,
            'vel_error': vel_error,
            'pos_rel_error': pos_rel_error,
            'vel_rel_error': vel_rel_error,
        }

        if verbose:
            print(f"Frame {self.frame} (t={self.time_elapsed:.4f}s):")
            print(f"  Centroid Y: sim={sim_centroid[1]:.6f}, gt={gt_pos[1]:.6f}, err={pos_error:.2e}")
            print(f"  Velocity Y: sim={sim_velocity[1]:.6f}, gt={gt_vel[1]:.6f}, err={vel_error:.2e}")

        self.frame_results.append(result)
        return result


def run_freefall_test(demo='eight_E_freefall', frames=5, initial_vy=-1.0,
                      grad_tol=1e-6, iter_max=100, inversion_method='oneway_gj',
                      use_mas=False, verbose=True):
    """
    Run free-fall validation test.

    Args:
        demo: Demo configuration name
        frames: Number of frames to simulate
        initial_vy: Initial downward velocity
        grad_tol: Gradient tolerance for convergence
        iter_max: Maximum iterations per frame
        inversion_method: MAS block inversion method
        use_mas: If True, use MAS preconditioner; if False, use diagonal
        verbose: Print detailed output

    Returns:
        dict with test results
    """
    precond_type = "MAS" if use_mas else "Diagonal"
    print(f"\n{'='*70}")
    print(f"Eight E Free-Fall Validation Test ({precond_type} Preconditioner)")
    print(f"{'='*70}")
    print(f"Demo: {demo}")
    print(f"Frames: {frames}, Initial Vy: {initial_vy}")
    if use_mas:
        print(f"Inversion: {inversion_method}, grad_tol: {grad_tol}")
    else:
        print(f"grad_tol: {grad_tol}")
    print(f"{'='*70}\n")

    # Create solver
    solver = EightEFreeFallValidator(demo=demo, inversion_method=inversion_method, use_mas=use_mas)
    solver.iter_max = iter_max

    print(f"Material: E={solver.dict['E']}, nu={solver.dict['nu']}")
    print(f"dt={solver.dt}, gravity={solver.gravity}")

    # Set initial velocity
    solver.set_initial_velocity(initial_vy)

    # Run simulation
    print(f"\n[Running {frames} frames...]")
    total_iters = 0
    total_time = 0.0

    for f in range(frames):
        iters, elapsed = solver.step(verbose=verbose, grad_tol=grad_tol)
        result = solver.validate_frame(verbose=verbose)
        total_iters += iters
        total_time += elapsed

    # Summary
    print(f"\n{'='*70}")
    print(f"Test Summary")
    print(f"{'='*70}")

    # Compute final errors
    max_pos_error = max(r['pos_error'] for r in solver.frame_results)
    max_vel_error = max(r['vel_error'] for r in solver.frame_results)
    avg_pos_error = np.mean([r['pos_error'] for r in solver.frame_results])
    avg_vel_error = np.mean([r['vel_error'] for r in solver.frame_results])

    print(f"Position Error: max={max_pos_error:.2e}, avg={avg_pos_error:.2e}")
    print(f"Velocity Error: max={max_vel_error:.2e}, avg={avg_vel_error:.2e}")
    print(f"Total iterations: {total_iters}, Avg per frame: {total_iters/frames:.1f}")
    print(f"Total time: {total_time:.2f}ms, Avg per frame: {total_time/frames:.2f}ms")

    # Pass/Fail criteria
    # For a rigid-body-like motion, errors should be very small
    # Allow some tolerance for elastic deformation
    PASS_THRESHOLD_POS = 1e-2  # 1cm position error
    PASS_THRESHOLD_VEL = 1e-1  # 0.1 m/s velocity error

    passed = max_pos_error < PASS_THRESHOLD_POS and max_vel_error < PASS_THRESHOLD_VEL

    if passed:
        print(f"\n[PASSED] Solver produces correct update direction")
    else:
        print(f"\n[FAILED] Errors exceed threshold")
        print(f"  Position threshold: {PASS_THRESHOLD_POS}, actual: {max_pos_error:.2e}")
        print(f"  Velocity threshold: {PASS_THRESHOLD_VEL}, actual: {max_vel_error:.2e}")

    print(f"{'='*70}\n")

    return {
        'demo': demo,
        'preconditioner': precond_type,
        'passed': passed,
        'max_pos_error': max_pos_error,
        'max_vel_error': max_vel_error,
        'avg_pos_error': avg_pos_error,
        'avg_vel_error': avg_vel_error,
        'total_iters': total_iters,
        'total_time_ms': total_time,
        'avg_time_per_frame_ms': total_time / frames,
        'avg_iters_per_frame': total_iters / frames,
        'frame_results': solver.frame_results,
    }


def run_comparison_test(frames=5, initial_vy=-1.0, grad_tol=1e-6, iter_max=100):
    """
    Compare MAS vs diagonal preconditioner performance.
    """
    print(f"\n{'='*70}")
    print(f"Eight E Free-Fall: MAS vs Diagonal Comparison")
    print(f"{'='*70}\n")

    # Run diagonal preconditioner
    print("Running Diagonal Preconditioner...")
    diag_result = run_freefall_test(
        demo='eight_E_freefall',
        frames=frames,
        initial_vy=initial_vy,
        grad_tol=grad_tol,
        iter_max=iter_max,
        use_mas=False,
        verbose=False
    )

    # Run MAS preconditioner
    print("\nRunning MAS Preconditioner...")
    mas_result = run_freefall_test(
        demo='eight_E_freefall_mas',
        frames=frames,
        initial_vy=initial_vy,
        grad_tol=grad_tol,
        iter_max=iter_max,
        use_mas=True,
        verbose=False
    )

    # Comparison summary
    print(f"\n{'='*70}")
    print(f"Comparison Summary")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Diagonal':<20} {'MAS':<20} {'Speedup':<10}")
    print('-' * 80)
    print(f"{'Avg time per frame (ms)':<30} {diag_result['avg_time_per_frame_ms']:<20.2f} {mas_result['avg_time_per_frame_ms']:<20.2f} {diag_result['avg_time_per_frame_ms']/mas_result['avg_time_per_frame_ms']:<10.2f}x")
    print(f"{'Avg iterations per frame':<30} {diag_result['avg_iters_per_frame']:<20.1f} {mas_result['avg_iters_per_frame']:<20.1f} {diag_result['avg_iters_per_frame']/mas_result['avg_iters_per_frame']:<10.2f}x")
    print(f"{'Max position error':<30} {diag_result['max_pos_error']:<20.2e} {mas_result['max_pos_error']:<20.2e}")
    print(f"{'Max velocity error':<30} {diag_result['max_vel_error']:<20.2e} {mas_result['max_vel_error']:<20.2e}")
    print(f"{'Passed':<30} {str(diag_result['passed']):<20} {str(mas_result['passed']):<20}")
    print(f"{'='*70}\n")

    return {
        'diagonal': diag_result,
        'mas': mas_result,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eight E Free-Fall Validation Test')
    parser.add_argument('--frames', type=int, default=5, help='Number of frames')
    parser.add_argument('--vy', type=float, default=-1.0, help='Initial downward velocity')
    parser.add_argument('--grad_tol', type=float, default=1e-6, help='Gradient tolerance')
    parser.add_argument('--iter_max', type=int, default=100, help='Max iterations per frame')
    parser.add_argument('--inversion', type=str, default='oneway_gj',
                        choices=['cholesky', 'gauss_jordan', 'oneway_gj', 'incomplete'],
                        help='MAS block inversion method')
    parser.add_argument('--use-mas', action='store_true', help='Use MAS preconditioner')
    parser.add_argument('--compare', action='store_true', help='Compare MAS vs diagonal')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    args = parser.parse_args()

    ti.init(arch=ti.gpu, default_fp=ti.f32)

    if args.compare:
        run_comparison_test(
            frames=args.frames,
            initial_vy=args.vy,
            grad_tol=args.grad_tol,
            iter_max=args.iter_max
        )
    else:
        demo = 'eight_E_freefall_mas' if args.use_mas else 'eight_E_freefall'
        run_freefall_test(
            demo=demo,
            frames=args.frames,
            initial_vy=args.vy,
            grad_tol=args.grad_tol,
            iter_max=args.iter_max,
            inversion_method=args.inversion,
            use_mas=args.use_mas,
            verbose=not args.quiet
        )
