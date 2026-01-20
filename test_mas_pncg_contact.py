#!/usr/bin/env python
"""Test script for MAS-PNCG solver with contact."""

import taichi as ti
import time
import sys

# Initialize Taichi
ti.init(arch=ti.gpu, default_fp=ti.f32, offline_cache=True, offline_cache_file_path='.taichi_cache')
print('Taichi initialized', flush=True)

from algorithm.mas_pncg_solver import MASPNCGSolver

print('Creating solver...', flush=True)
solver = MASPNCGSolver(demo='eight_E_drop_demo_contact')

print('\nRunning 2 frames...\n', flush=True)
for f in range(2):
    t0 = time.perf_counter()
    iters = solver.step(verbose=True)
    t1 = time.perf_counter()
    print(f'\n>>> Frame {f}: {iters} iters, {(t1-t0)*1000:.2f}ms\n', flush=True)

print('Test completed!', flush=True)
