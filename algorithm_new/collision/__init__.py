"""
Collision detection module for algorithm_new.

Provides BVH-based collision detection implementing the CollisionDetector protocol.

Two detector implementations:
- BVHCollisionDetector: Original with separate AABB fields
- BVHCollisionDetectorFinal: Optimized with packed AABB (vec6) storage
"""

from .registry import CollisionDetectorRegistry
from .contact_pair import ContactPairStorage
from .bvh_detector import BVHCollisionDetector, SurfaceData
from .bvh_detector_final import BVHCollisionDetectorFinal

__all__ = [
    'CollisionDetectorRegistry',
    'ContactPairStorage',
    'BVHCollisionDetector',
    'BVHCollisionDetectorFinal',
    'SurfaceData',
]
