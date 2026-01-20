"""
Collision detection module for algorithm_new.

Provides BVH-based collision detection implementing the CollisionDetector protocol.
"""

from .registry import CollisionDetectorRegistry
from .contact_pair import ContactPairStorage
from .bvh_detector import BVHCollisionDetector, SurfaceData

__all__ = [
    'CollisionDetectorRegistry',
    'ContactPairStorage',
    'BVHCollisionDetector',
    'SurfaceData',
]
