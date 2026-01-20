"""
Contact Filter Module for Two-Radius Collision Detection Strategy.

This module implements a contact filter that uses a larger detection radius
(detection_dHat = 5 * dHat) for caching collision pairs, but filters them
to only keep contacts within the active barrier radius (active_dHat = dHat).

Usage:
    from algorithm.contact_filter import ContactFilter

    # Setup
    contact_filter = ContactFilter(max_contacts=2**21)
    detection_dHat = contact_filter.configure(dHat, multiplier=5.0)

    # Configure collision detection with larger radius
    deformer.set_detection_dHat(detection_dHat)

    # During simulation loop:
    deformer.find_cnts()  # Detects with detection_dHat (5*dHat)

    # Filter to active contacts
    contact_filter.filter_contacts(
        deformer.contact_pairs,
        deformer.n_contacts[None]
    )

    # Use filtered contacts for barrier computation
    n_active = contact_filter.n_filtered[None]
"""

import taichi as ti


@ti.data_oriented
class ContactFilter:
    """
    Contact filter for two-radius collision detection strategy.

    Uses detection_dHat (default: 5*dHat) for caching collision pairs,
    then filters to active_dHat (dHat) for barrier computation.

    This enables:
    1. Reduced BVH traversal frequency (larger cache validity window)
    2. Pre-filtering of irrelevant collisions before barrier computation
    3. Separation of detection and activation thresholds

    Attributes:
        detection_dHat: Large radius for caching collision pairs
        active_dHat: Actual barrier activation threshold
        filtered_contacts: Array of contacts within active_dHat
        n_filtered: Number of active contacts after filtering
    """

    def __init__(self, max_contacts: int = 2**21):
        """
        Initialize contact filter.

        Args:
            max_contacts: Maximum number of contacts to store
        """
        self.max_contacts = max_contacts

        # Distance thresholds
        self.detection_dHat = ti.field(dtype=ti.f32, shape=())
        self.active_dHat = ti.field(dtype=ti.f32, shape=())

        # Contact pair struct (matches collision_detection_bvh.py)
        self.pair = ti.types.struct(
            a=ti.types.vector(4, ti.u32),   # ids (vertex indices)
            b=float,                         # dist (distance)
            c=ti.types.vector(4, float),    # cord (barycentric coordinates)
            d=ti.types.vector(3, float)     # t (direction vector)
        )

        # Filtered contact storage
        self.filtered_contacts = self.pair.field(shape=max_contacts)
        self.n_filtered = ti.field(dtype=ti.i32, shape=())

        # Statistics
        self.n_cached = ti.field(dtype=ti.i32, shape=())

    def configure(self, active_dHat: float, multiplier: float = 5.0) -> float:
        """
        Configure detection and active thresholds.

        Args:
            active_dHat: The actual barrier threshold (dHat)
            multiplier: Multiplier for detection radius (default: 5.0)

        Returns:
            detection_dHat: The configured detection threshold for BVH
        """
        detection_dHat = active_dHat * multiplier
        self.detection_dHat[None] = detection_dHat
        self.active_dHat[None] = active_dHat
        return detection_dHat

    def get_detection_dHat(self) -> float:
        """Get the current detection threshold."""
        return self.detection_dHat[None]

    def get_active_dHat(self) -> float:
        """Get the current active threshold."""
        return self.active_dHat[None]

    @ti.kernel
    def filter_contacts(self, cached_contacts: ti.template(), n_cached: ti.i32):
        """
        Filter cached contacts to only those within active_dHat.

        Args:
            cached_contacts: Contact pairs from collision_detection_bvh
            n_cached: Number of cached contacts
        """
        self.n_filtered[None] = 0
        self.n_cached[None] = n_cached
        active_dHat = self.active_dHat[None]

        for idx in range(n_cached):
            pair = cached_contacts[idx]
            dist = pair.b

            # Filter: only keep contacts within active threshold
            if dist < active_dHat:
                filtered_idx = ti.atomic_add(self.n_filtered[None], 1)
                if filtered_idx < self.max_contacts:
                    self.filtered_contacts[filtered_idx] = pair

    @ti.kernel
    def filter_contacts_with_distance_check(
        self,
        cached_contacts: ti.template(),
        n_cached: ti.i32,
        min_dist: float
    ):
        """
        Filter cached contacts with additional minimum distance check.

        Args:
            cached_contacts: Contact pairs from collision_detection_bvh
            n_cached: Number of cached contacts
            min_dist: Minimum distance threshold (e.g., SMALL_NUM)
        """
        self.n_filtered[None] = 0
        self.n_cached[None] = n_cached
        active_dHat = self.active_dHat[None]

        for idx in range(n_cached):
            pair = cached_contacts[idx]
            dist = pair.b

            # Filter: within active threshold and above minimum
            if dist < active_dHat and dist > min_dist:
                filtered_idx = ti.atomic_add(self.n_filtered[None], 1)
                if filtered_idx < self.max_contacts:
                    self.filtered_contacts[filtered_idx] = pair

    def get_stats(self) -> dict:
        """
        Get filtering statistics.

        Returns:
            Dictionary with cached count, filtered count, and filter ratio
        """
        n_cached = self.n_cached[None]
        n_filtered = self.n_filtered[None]
        ratio = n_filtered / n_cached if n_cached > 0 else 0.0
        return {
            'n_cached': n_cached,
            'n_filtered': n_filtered,
            'filter_ratio': ratio
        }

    def print_stats(self):
        """Print filtering statistics."""
        stats = self.get_stats()
        print(f"Contact Filter: cached={stats['n_cached']}, "
              f"active={stats['n_filtered']}, "
              f"ratio={stats['filter_ratio']:.2%}")
