"""
Contact handling module for IPC, GCP and ground contact.

Provides barrier functions, contact handlers, and CCD step size computation.
"""

from .barrier import LogBarrier, CubicBarrier
from .ipc_handler import IPCContactHandler, compute_adaptive_kappa
from .ground_handler import GroundContactHandler
from .ccd_step_size import CCDStepSizeComputer
from .gcp import GCPConfig, GCPBarrier, GCPContactHandler

__all__ = [
    # IPC Barrier functions
    'LogBarrier',
    'CubicBarrier',
    # GCP
    'GCPConfig',
    'GCPBarrier',
    'GCPContactHandler',
    # Contact handlers
    'IPCContactHandler',
    'GroundContactHandler',
    # Step size computation
    'CCDStepSizeComputer',
    # Utilities
    'compute_adaptive_kappa',
]
