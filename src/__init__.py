"""
SAFE-Gate: Safety-first Abstention-enabled Formal triage Engine with parallel GATEs

A formally verified clinical triage architecture for dizziness and vertigo presentations
in emergency departments.

Authors: Chatchai Tritham, Chakkrit Snae Namahoot
Institution: Naresuan University, Thailand
"""

from .constants import FRAMEWORK_NAME, PACKAGE_VERSION

__version__ = PACKAGE_VERSION
__author__ = "Chatchai Tritham, Chakkrit Snae Namahoot"
__email__ = "chakkrits@nu.ac.th"

from .safegate import SAFEGate

__all__ = ['SAFEGate', 'FRAMEWORK_NAME']
