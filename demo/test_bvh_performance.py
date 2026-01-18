"""
BVH Performance Test Demo
Evaluates timing for each part of BVH build and refit operations.
"""
import sys
sys.path.insert(0, '..')

import taichi as ti
import numpy as np
import time
from taichi.algorithms import parallel_sort

ti.init(arch=ti.gpu)


@ti.data_oriented
class LBVH_Benchmark:
    """
    LBVH with detailed timing for each step.
    """

    def __init__(self, max_primitives: int):
        self.max_primitives = max_primitives
        self.num_nodes = 2 * max_primitives - 1

        # AABB bounding volumes
        self.bv_lower = ti.Vector.field(3, dtype=ti.f32, shape=self.num_nodes)
        self.bv_upper = ti.Vector.field(3, dtype=ti.f32, shape=self.num_nodes)

        # Node structure
        self.parent_idx = ti.field(dtype=ti.u32, shape=self.num_nodes)
        self.left_idx = ti.field(dtype=ti.u32, shape=self.num_nodes)
        self.right_idx = ti.field(dtype=ti.u32, shape=self.num_nodes)
        self.element_idx = ti.field(dtype=ti.u32, shape=self.num_nodes)

        # Morton codes and indices
        self.morton_codes = ti.field(dtype=ti.u64, shape=max_primitives)
        self.sorted_indices = ti.field(dtype=ti.u32, shape=max_primitives)

        # Flags for bottom-up AABB computation
        self.flags = ti.field(dtype=ti.u32, shape=max_primitives)

        # Scene bounding box
        self.scene_lower = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.scene_upper = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.num_primitives = ti.field(dtype=ti.i32, shape=())
        self.INVALID = 0xFFFFFFFF
        self.tree_built = False

    @ti.func
    def expand_bits(self, v: ti.u32) -> ti.u32:
        v = (v * ti.u32(0x00010001)) & ti.u32(0xFF0000FF)
        v = (v * ti.u32(0x00000101)) & ti.u32(0x0F00F00F)
        v = (v * ti.u32(0x00000011)) & ti.u32(0xC30C30C3)
        v = (v * ti.u32(0x00000005)) & ti.u32(0x49249249)
        return v

    @ti.func
    def morton_code_3d(self, x: ti.f32, y: ti.f32, z: ti.f32) -> ti.u32:
        resolution = 1024.0
        x = ti.min(ti.max(x * resolution, 0.0), resolution - 1.0)
        y = ti.min(ti.max(y * resolution, 0.0), resolution - 1.0)
        z = ti.min(ti.max(z * resolution, 0.0), resolution - 1.0)
        xx = self.expand_bits(ti.cast(x, ti.u32))
        yy = self.expand_bits(ti.cast(y, ti.u32))
        zz = self.expand_bits(ti.cast(z, ti.u32))
        return (xx << 2) | (yy << 1) | zz

    @ti.func
    def aabb_merge(self, idx1: ti.i32, idx2: ti.i32, dst: ti.i32):
        self.bv_lower[dst] = ti.min(self.bv_lower[idx1], self.bv_lower[idx2])
        self.bv_upper[dst] = ti.max(self.bv_upper[idx1], self.bv_upper[idx2])

    @ti.func
    def common_upper_bits(self, lhs: ti.u64, rhs: ti.u64) -> ti.i32:
        xor_val = lhs ^ rhs
        count = 0
        if xor_val == 0:
            count = 64
        else:
            # Binary search for leading zeros (faster than 64-iteration loop)
            if (xor_val >> 32) == 0: count += 32; xor_val <<= 32
            if (xor_val >> 48) == 0: count += 16; xor_val <<= 16
            if (xor_val >> 56) == 0: count += 8;  xor_val <<= 8
            if (xor_val >> 60) == 0: count += 4;  xor_val <<= 4
            if (xor_val >> 62) == 0: count += 2;  xor_val <<= 2
            if (xor_val >> 63) == 0: count += 1
        return count

    @ti.func
    def determine_range(self, idx: ti.i32, num_leaves: ti.i32) -> ti.math.ivec2:
        first = 0
        last = num_leaves - 1

        if idx != 0:
            self_code = self.morton_codes[idx]
            L_delta = self.common_upper_bits(self_code, self.morton_codes[idx - 1])
            R_delta = self.common_upper_bits(self_code, self.morton_codes[idx + 1])

            d = 1 if R_delta > L_delta else -1
            delta_min = ti.min(L_delta, R_delta)

            l_max = 2
            i_tmp = idx + d * l_max
            delta = -1
            if 0 <= i_tmp < num_leaves:
                delta = self.common_upper_bits(self_code, self.morton_codes[i_tmp])

            while delta > delta_min:
                l_max <<= 1
                i_tmp = idx + d * l_max
                delta = -1
                if 0 <= i_tmp < num_leaves:
                    delta = self.common_upper_bits(self_code, self.morton_codes[i_tmp])

            l = 0
            t = l_max >> 1
            while t > 0:
                i_tmp = idx + (l + t) * d
                delta = -1
                if 0 <= i_tmp < num_leaves:
                    delta = self.common_upper_bits(self_code, self.morton_codes[i_tmp])
                if delta > delta_min:
                    l += t
                t >>= 1

            jdx = idx + l * d
            first = ti.min(idx, jdx)
            last = ti.max(idx, jdx)

        return ti.math.ivec2(first, last)

    @ti.func
    def find_split(self, first: ti.i32, last: ti.i32) -> ti.i32:
        first_code = self.morton_codes[first]
        last_code = self.morton_codes[last]
        split = (first + last) >> 1

        if first_code != last_code:
            delta_node = self.common_upper_bits(first_code, last_code)
            split = first
            stride = last - first
            while stride > 1:
                stride = (stride + 1) >> 1
                middle = split + stride
                if middle < last:
                    delta = self.common_upper_bits(first_code, self.morton_codes[middle])
                    if delta > delta_node:
                        split = middle
        return split

    # ========== BUILD KERNELS ==========

    @ti.kernel
    def compute_leaf_aabbs(self, vertices: ti.template(), triangles: ti.template(), n: ti.i32):
        self.num_primitives[None] = n
        for i in range(n):
            t0, t1, t2 = triangles[i, 0], triangles[i, 1], triangles[i, 2]
            v0, v1, v2 = vertices[t0], vertices[t1], vertices[t2]
            leaf_idx = i + n - 1
            self.bv_lower[leaf_idx] = ti.min(ti.min(v0, v1), v2)
            self.bv_upper[leaf_idx] = ti.max(ti.max(v0, v1), v2)

    @ti.kernel
    def compute_scene_aabb(self, n: ti.i32):
        scene_min = ti.Vector([1e32, 1e32, 1e32])
        scene_max = ti.Vector([-1e32, -1e32, -1e32])
        for i in range(n):
            leaf_idx = i + n - 1
            ti.atomic_min(scene_min, self.bv_lower[leaf_idx])
            ti.atomic_max(scene_max, self.bv_upper[leaf_idx])
        self.scene_lower[None] = scene_min
        self.scene_upper[None] = scene_max

    @ti.kernel
    def compute_morton_codes(self, n: ti.i32):
        scene_lower = self.scene_lower[None]
        scene_size = ti.max(self.scene_upper[None] - scene_lower, ti.Vector([1e-10, 1e-10, 1e-10]))
        for i in range(n):
            leaf_idx = i + n - 1
            center = (self.bv_lower[leaf_idx] + self.bv_upper[leaf_idx]) * 0.5
            normalized = (center - scene_lower) / scene_size
            mc32 = self.morton_code_3d(normalized[0], normalized[1], normalized[2])
            self.morton_codes[i] = (ti.cast(mc32, ti.u64) << 32) | ti.cast(i, ti.u64)
            self.sorted_indices[i] = ti.cast(i, ti.u32)

    @ti.kernel
    def _prepare_sort(self, n: ti.i32):
        max_prims = self.max_primitives
        for i in range(max_prims):
            self.sorted_indices[i] = ti.cast(i, ti.u32)
            if i >= n:
                self.morton_codes[i] = ti.u64(0xFFFFFFFFFFFFFFFF)

    def sort_morton_codes(self, n: int):
        self._prepare_sort(n)
        parallel_sort(self.morton_codes, self.sorted_indices)

    @ti.kernel
    def init_leaf_nodes(self, n: ti.i32):
        INVALID = ti.u32(0xFFFFFFFF)
        for i in range(n):
            if i < n - 1:
                self.left_idx[i] = INVALID
                self.right_idx[i] = INVALID
                self.parent_idx[i] = INVALID
                self.element_idx[i] = INVALID
            leaf_idx = i + n - 1
            self.left_idx[leaf_idx] = INVALID
            self.right_idx[leaf_idx] = INVALID
            self.parent_idx[leaf_idx] = INVALID
            self.element_idx[leaf_idx] = self.sorted_indices[i]

    @ti.kernel
    def build_internal_nodes(self, n: ti.i32):
        for i in range(n - 1):
            range_ij = self.determine_range(i, n)
            first, last = range_ij[0], range_ij[1]
            gamma = self.find_split(first, last)

            left_child = gamma
            right_child = gamma + 1
            if ti.min(first, last) == gamma:
                left_child += n - 1
            if ti.max(first, last) == gamma + 1:
                right_child += n - 1

            self.left_idx[i] = left_child
            self.right_idx[i] = right_child
            self.parent_idx[left_child] = i
            self.parent_idx[right_child] = i

    @ti.kernel
    def compute_internal_aabbs(self, n: ti.i32):
        INVALID = ti.u32(0xFFFFFFFF)
        for i in range(n - 1):
            self.flags[i] = INVALID

        for i in range(n):
            leaf_idx = i + n - 1
            parent = self.parent_idx[leaf_idx]
            while parent != INVALID:
                old = ti.atomic_and(self.flags[parent], ti.u32(0))
                if old == INVALID:
                    break
                left = self.left_idx[parent]
                right = self.right_idx[parent]
                self.aabb_merge(ti.cast(left, ti.i32), ti.cast(right, ti.i32), ti.cast(parent, ti.i32))
                ti.simt.block.mem_sync()
                parent = self.parent_idx[parent]

    @ti.kernel
    def reorder_leaf_aabbs(self, n: ti.i32):
        for i in range(n):
            orig_idx = n - 1 + i
            self.bv_lower[i] = self.bv_lower[orig_idx]
            self.bv_upper[i] = self.bv_upper[orig_idx]
        ti.sync()
        for i in range(n):
            src_idx = ti.cast(self.sorted_indices[i], ti.i32)
            dst_idx = n - 1 + i
            self.bv_lower[dst_idx] = self.bv_lower[src_idx]
            self.bv_upper[dst_idx] = self.bv_upper[src_idx]

    # ========== REFIT KERNELS ==========

    @ti.kernel
    def refit_leaf_aabbs(self, vertices: ti.template(), triangles: ti.template(), n: ti.i32):
        for i in range(n):
            leaf_idx = i + n - 1
            tri_idx = ti.cast(self.element_idx[leaf_idx], ti.i32)
            t0, t1, t2 = triangles[tri_idx, 0], triangles[tri_idx, 1], triangles[tri_idx, 2]
            v0, v1, v2 = vertices[t0], vertices[t1], vertices[t2]
            self.bv_lower[leaf_idx] = ti.min(ti.min(v0, v1), v2)
            self.bv_upper[leaf_idx] = ti.max(ti.max(v0, v1), v2)

    # ========== TIMED BUILD/REFIT ==========

    def build_timed(self, vertices, triangles, n: int):
        """Build BVH with timing for each step."""
        timings = {}

        ti.sync()
        t0 = time.perf_counter()
        self.compute_leaf_aabbs(vertices, triangles, n)
        ti.sync()
        t1 = time.perf_counter()
        timings['1_leaf_aabbs'] = (t1 - t0) * 1000

        self.compute_scene_aabb(n)
        ti.sync()
        t2 = time.perf_counter()
        timings['2_scene_aabb'] = (t2 - t1) * 1000

        self.compute_morton_codes(n)
        ti.sync()
        t3 = time.perf_counter()
        timings['3_morton_codes'] = (t3 - t2) * 1000

        self.sort_morton_codes(n)
        ti.sync()
        t4 = time.perf_counter()
        timings['4_sort'] = (t4 - t3) * 1000

        self.reorder_leaf_aabbs(n)
        ti.sync()
        t5 = time.perf_counter()
        timings['5_reorder_aabbs'] = (t5 - t4) * 1000

        self.init_leaf_nodes(n)
        ti.sync()
        t6 = time.perf_counter()
        timings['6_init_leaves'] = (t6 - t5) * 1000

        self.build_internal_nodes(n)
        ti.sync()
        t7 = time.perf_counter()
        timings['7_build_internal'] = (t7 - t6) * 1000

        self.compute_internal_aabbs(n)
        ti.sync()
        t8 = time.perf_counter()
        timings['8_internal_aabbs'] = (t8 - t7) * 1000

        timings['total'] = (t8 - t0) * 1000
        self.tree_built = True

        return timings

    def refit_timed(self, vertices, triangles, n: int):
        """Refit BVH with timing for each step."""
        if not self.tree_built:
            return self.build_timed(vertices, triangles, n)

        timings = {}

        ti.sync()
        t0 = time.perf_counter()
        self.refit_leaf_aabbs(vertices, triangles, n)
        ti.sync()
        t1 = time.perf_counter()
        timings['1_refit_leaf_aabbs'] = (t1 - t0) * 1000

        self.compute_internal_aabbs(n)
        ti.sync()
        t2 = time.perf_counter()
        timings['2_propagate_aabbs'] = (t2 - t1) * 1000

        timings['total'] = (t2 - t0) * 1000

        return timings


# ========== TEST DATA GENERATION ==========

def create_test_mesh(n_triangles: int):
    """Create test mesh with random triangles."""
    n_verts = n_triangles * 3
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)
    triangles = ti.field(dtype=ti.i32, shape=(n_triangles, 3))

    @ti.kernel
    def init():
        for i in range(n_verts):
            vertices[i] = ti.Vector([ti.random(), ti.random(), ti.random()])
        for i in range(n_triangles):
            triangles[i, 0] = i * 3
            triangles[i, 1] = i * 3 + 1
            triangles[i, 2] = i * 3 + 2

    init()
    return vertices, triangles


@ti.kernel
def perturb_vertices(vertices: ti.template(), n_verts: int, amount: float):
    """Simulate deformation by perturbing vertices."""
    for i in range(n_verts):
        vertices[i] += ti.Vector([
            (ti.random() - 0.5) * amount,
            (ti.random() - 0.5) * amount,
            (ti.random() - 0.5) * amount
        ])


# ========== BENCHMARK ==========

def run_benchmark(n_triangles: int, iterations: int = 10, warmup: int = 3):
    """Run benchmark for given triangle count."""
    print(f"\n{'='*60}")
    print(f"Benchmarking with {n_triangles:,} triangles")
    print(f"{'='*60}")

    vertices, triangles = create_test_mesh(n_triangles)
    bvh = LBVH_Benchmark(n_triangles + 100)

    # Warmup
    for _ in range(warmup):
        bvh.tree_built = False
        bvh.build_timed(vertices, triangles, n_triangles)
        perturb_vertices(vertices, n_triangles * 3, 0.01)
        bvh.refit_timed(vertices, triangles, n_triangles)

    # Benchmark BUILD
    build_times = []
    build_breakdown = {}
    for _ in range(iterations):
        bvh.tree_built = False
        timings = bvh.build_timed(vertices, triangles, n_triangles)
        build_times.append(timings['total'])
        for k, v in timings.items():
            if k not in build_breakdown:
                build_breakdown[k] = []
            build_breakdown[k].append(v)

    # Benchmark REFIT
    refit_times = []
    refit_breakdown = {}
    for _ in range(iterations):
        perturb_vertices(vertices, n_triangles * 3, 0.01)
        timings = bvh.refit_timed(vertices, triangles, n_triangles)
        refit_times.append(timings['total'])
        for k, v in timings.items():
            if k not in refit_breakdown:
                refit_breakdown[k] = []
            refit_breakdown[k].append(v)

    # Print results
    print(f"\n--- BUILD Timing Breakdown (avg of {iterations} runs) ---")
    for k in sorted(build_breakdown.keys()):
        if k != 'total':
            avg = np.mean(build_breakdown[k])
            std = np.std(build_breakdown[k])
            pct = avg / np.mean(build_breakdown['total']) * 100
            print(f"  {k:25s}: {avg:8.3f} ms ± {std:5.3f} ({pct:5.1f}%)")
    print(f"  {'TOTAL':25s}: {np.mean(build_times):8.3f} ms ± {np.std(build_times):5.3f}")

    print(f"\n--- REFIT Timing Breakdown (avg of {iterations} runs) ---")
    for k in sorted(refit_breakdown.keys()):
        if k != 'total':
            avg = np.mean(refit_breakdown[k])
            std = np.std(refit_breakdown[k])
            pct = avg / np.mean(refit_breakdown['total']) * 100
            print(f"  {k:25s}: {avg:8.3f} ms ± {std:5.3f} ({pct:5.1f}%)")
    print(f"  {'TOTAL':25s}: {np.mean(refit_times):8.3f} ms ± {np.std(refit_times):5.3f}")

    speedup = np.mean(build_times) / np.mean(refit_times)
    print(f"\n  Build/Refit Speedup: {speedup:.2f}x")

    return {
        'n_triangles': n_triangles,
        'build_avg': np.mean(build_times),
        'build_std': np.std(build_times),
        'refit_avg': np.mean(refit_times),
        'refit_std': np.std(refit_times),
        'speedup': speedup
    }


def main():
    print("=" * 60)
    print("BVH Performance Test")
    print("=" * 60)

    # Test with different mesh sizes
    test_sizes = [1000, 5000, 10000, 50000, 100000]
    results = []

    for n in test_sizes:
        result = run_benchmark(n, iterations=10, warmup=3)
        results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Triangles':>12} | {'Build (ms)':>15} | {'Refit (ms)':>15} | {'Speedup':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['n_triangles']:>12,} | {r['build_avg']:>10.3f} ± {r['build_std']:>4.2f} | "
              f"{r['refit_avg']:>10.3f} ± {r['refit_std']:>4.2f} | {r['speedup']:>9.2f}x")

    print("\nKey findings:")
    print("- Refit only updates leaf AABBs and propagates changes (no sorting/rebuild)")
    print("- Refit speedup increases with mesh complexity")


if __name__ == '__main__':
    main()
