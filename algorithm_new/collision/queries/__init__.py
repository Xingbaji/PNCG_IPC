"""
Distance computation and CCD query functions for collision detection.
"""

from .distance import (
    dist3D_Segment_to_Segment,
    dist3D_Point_Triangle,
    point_point_distance,
    point_edge_distance,
    point_triangle_distance,
    edge_edge_distance,
    point_triangle_distance_unclassified,
    edge_edge_distance_unclassified,
)

from .ccd import (
    point_triangle_ccd_broadphase,
    edge_edge_ccd_broadphase,
    point_triangle_ccd,
    edge_edge_ccd,
    point_triangle_ccd_lower_bound,
    edge_edge_ccd_lower_bound,
)

from .intersection import (
    segment_triangle_intersect_cramer,
)

__all__ = [
    # Distance functions
    'dist3D_Segment_to_Segment',
    'dist3D_Point_Triangle',
    'point_point_distance',
    'point_edge_distance',
    'point_triangle_distance',
    'edge_edge_distance',
    'point_triangle_distance_unclassified',
    'edge_edge_distance_unclassified',
    # CCD functions
    'point_triangle_ccd_broadphase',
    'edge_edge_ccd_broadphase',
    'point_triangle_ccd',
    'edge_edge_ccd',
    'point_triangle_ccd_lower_bound',
    'edge_edge_ccd_lower_bound',
    # Intersection functions
    'segment_triangle_intersect_cramer',
]
