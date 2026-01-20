"""
Contact pair storage for collision detection results.

Provides GPU-friendly storage for detected contact pairs with atomic insertion.
"""

import taichi as ti
from ..core.precision import PrecisionType, PrecisionMixin, get_precision_config


@ti.data_oriented
class ContactPairStorage(PrecisionMixin):
    """
    GPU-friendly contact pair storage with atomic insertion.

    Stores contact pairs detected during collision detection, including:
    - Vertex indices (4 indices for PT or EE contacts)
    - Distance
    - Barycentric coordinates
    - Contact direction vector

    Usage:
        storage = ContactPairStorage(max_contacts=2**20, precision='f32')
        storage.reset()
        # ... collision detection adds pairs ...
        n = storage.count
        pairs = storage.contact_pairs
    """

    def __init__(self, max_contacts: int = 2**20, precision: PrecisionType = 'f32'):
        """
        Initialize contact pair storage.

        Args:
            max_contacts: Maximum number of contact pairs to store
            precision: Float precision ('f32' or 'f64')
        """
        self.init_precision(precision)
        self.MAX_C = max_contacts

        # Get float type for precision-aware struct
        float_type = self.cfg.float_type

        # Define contact pair struct
        # a: 4 vertex indices (u32)
        # b: distance (float)
        # c: barycentric coordinates (4 floats)
        # d: contact direction vector (3 floats)
        self.pair = ti.types.struct(
            a=ti.types.vector(4, ti.u32),    # ids: vertex indices
            b=float_type,                     # dist: distance
            c=ti.types.vector(4, float_type), # cord: barycentric coordinates
            d=ti.types.vector(3, float_type)  # t: contact direction
        )

        # Allocate contact pair storage
        self.contact_pairs = self.pair.field(shape=max_contacts)

        # Atomic counter for number of contacts
        self._n_contacts = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def _reset_kernel(self):
        """Reset the contact counter."""
        self._n_contacts[None] = 0

    def reset(self):
        """Reset storage for new frame."""
        self._reset_kernel()

    @ti.func
    def add_pair(
        self,
        ids: ti.types.vector(4, ti.u32),
        dist: ti.template(),
        cord: ti.template(),
        direction: ti.template()
    ) -> ti.i32:
        """
        Atomically add a contact pair.

        Args:
            ids: 4 vertex indices
            dist: Distance value
            cord: 4 barycentric coordinates
            direction: 3D contact direction vector

        Returns:
            Index where pair was stored, or -1 if storage full
        """
        idx = ti.atomic_add(self._n_contacts[None], 1)
        result = ti.cast(-1, ti.i32)
        if idx < self.MAX_C:
            self.contact_pairs[idx].a = ids
            self.contact_pairs[idx].b = dist
            self.contact_pairs[idx].c = cord
            self.contact_pairs[idx].d = direction
            result = idx
        else:
            # Storage full, decrement counter
            ti.atomic_sub(self._n_contacts[None], 1)
        return result

    @ti.func
    def add_pair_PT(
        self,
        p_idx: ti.u32,
        t0_idx: ti.u32,
        t1_idx: ti.u32,
        t2_idx: ti.u32,
        dist: ti.template(),
        cord: ti.template(),
        direction: ti.template()
    ) -> ti.i32:
        """
        Add a Point-Triangle contact pair.

        Args:
            p_idx: Point vertex index
            t0_idx, t1_idx, t2_idx: Triangle vertex indices
            dist: Distance value
            cord: Barycentric coordinates [w_p, w_t0, w_t1, w_t2]
            direction: Contact direction (normalized)

        Returns:
            Index where pair was stored, or -1 if storage full
        """
        ids = ti.Vector([p_idx, t0_idx, t1_idx, t2_idx], dt=ti.u32)
        return self.add_pair(ids, dist, cord, direction)

    @ti.func
    def add_pair_EE(
        self,
        a0_idx: ti.u32,
        a1_idx: ti.u32,
        b0_idx: ti.u32,
        b1_idx: ti.u32,
        dist: ti.template(),
        cord: ti.template(),
        direction: ti.template()
    ) -> ti.i32:
        """
        Add an Edge-Edge contact pair.

        Args:
            a0_idx, a1_idx: First edge vertex indices
            b0_idx, b1_idx: Second edge vertex indices
            dist: Distance value
            cord: Edge parameters [1-s, s, 1-t, t] where s,t are edge parameters
            direction: Contact direction (normalized)

        Returns:
            Index where pair was stored, or -1 if storage full
        """
        ids = ti.Vector([a0_idx, a1_idx, b0_idx, b1_idx], dt=ti.u32)
        return self.add_pair(ids, dist, cord, direction)

    @property
    def count(self) -> int:
        """Get the current number of contact pairs."""
        return min(self._n_contacts[None], self.MAX_C)

    @property
    def n_contacts(self) -> int:
        """Alias for count - number of active contacts."""
        return self.count

    def get_pair(self, idx: int):
        """
        Get a contact pair by index (for debugging/testing).

        Args:
            idx: Index of the contact pair

        Returns:
            Tuple of (ids, dist, cord, direction) as numpy arrays
        """
        import numpy as np
        if idx >= self.count:
            raise IndexError(f"Index {idx} out of range (count={self.count})")

        pair = self.contact_pairs[idx]
        return (
            pair.a.to_numpy(),
            float(pair.b),
            pair.c.to_numpy(),
            pair.d.to_numpy()
        )
