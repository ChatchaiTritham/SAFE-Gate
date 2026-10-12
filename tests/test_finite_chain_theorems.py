"""Fast exhaustive checks of the finite-chain theorems (scripts/verify_theorems.py)."""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import verify_theorems as vt  # noqa: E402


def test_quota_characterisation_converse():
    for n in (3, 4):
        r = vt.t1_converse(n, 6)
        assert r["monotone_equals_quota"] and r["monotone_maps"] == r["quota_maps"] == 7


def test_definition_equals_quota_for_theta_le_mu():
    conf = vt._CONF[::97]
    for tn, td, mn, md in [(1, 2, 1, 2), (1, 2, 2, 3)]:
        f = vt.acwcm_def(vt.PROFILES, conf, tn, td, mn, md)
        assert np.array_equal(f, vt.quota(vt.PROFILES, conf, mn, md))


def test_smooth_cnd_bound():
    r = vt.t2()
    assert r["U_equals_max(2,rank_rmin)"] and r["bound_attained_by_smooth_CND_operator"]
    assert r["two_gates"]["of_which_relax_a_floor_R2_to_R5"] == 0


def test_sugeno_smooth_and_cnd_degenerate():
    caps = list(vt.capacities(3))
    assert len(caps) == 5103
