# n_E Demo Performance Report

**Generated:** 2026-01-19 11:28:21
**Frames:** 5
**Demo:** eight_E_drop_demo_contact

## Algorithm Variants

| Version | Collision Method | Barrier Type | Description |
|---------|------------------|--------------|-------------|
| initial | bvh | log | Reference implementation (BVH + log barrier) |
| bvh | bvh | log | BVH collision (LBVH + Morton codes) |

## Performance Summary

| Version | Avg Frame (ms) | Avg Step (ms) | Avg Iterations | Avg FPS |
|---------|----------------|---------------|----------------|---------|
| initial | 1293.55 | 1196.49 | 15.0 | 0.77 |
| bvh | 1449.20 | 1354.90 | 15.0 | 0.69 |

## Detailed Timing Statistics

### Step Time (ms)

| Version | Mean | Std Dev | Min | Max |
|---------|------|---------|-----|-----|
| initial | 1196.49 | 1501.19 | 443.45 | 4198.86 |
| bvh | 1354.90 | 1824.33 | 439.47 | 5003.56 |

### Iterations per Frame

| Version | Mean | Std Dev | Min | Max |
|---------|------|---------|-----|-----|
| initial | 15.0 | 0.00 | 15 | 15 |
| bvh | 15.0 | 0.00 | 15 | 15 |

## Speedup Analysis


## Image Outputs

Images saved to:
- `./imgs/initial/frame_XXXXX.png`
- `./imgs/bvh/frame_XXXXX.png`

## Raw Data

Full timing data saved to: `./imgs/<version>/stats.json`

## Notes

- **initial**: Reference implementation from PNCG_IPC_init, uses BVH collision
- **bvh**: LBVH (Linear BVH) with Morton codes for spatial sorting
- **spatial_hash**: Uses Bresenham line algorithm for edge-grid rasterization
- **cubic_barrier**: Uses cubic barrier function: psi = -2k/(3*dHat) * (d - dHat)^3
- MAS preconditioner is NOT included in these tests
