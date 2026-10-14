"""Independent analytic and Lorentz-invariance checks of the femto observable."""

import copy
import unittest

import numpy as np
import pandas as pd

from pih3l.config import DEFAULTS
from pih3l.physics import (
    decode_its_cluster_sizes,
    derive,
    four_vector,
    invariant_mass,
    kstar,
)


def boost_x(vectors, beta):
    """Independent reference Lorentz boost; momenta ordered E, px, py, pz."""
    vectors = np.asarray(vectors, dtype=float)
    gamma = 1 / np.sqrt(1 - beta * beta)
    result = vectors.copy()
    result[:, 0] = gamma * (vectors[:, 0] + beta * vectors[:, 1])
    result[:, 1] = gamma * (vectors[:, 1] + beta * vectors[:, 0])
    return result


class PhysicsTests(unittest.TestCase):
    def test_four_vector_mass_shell(self):
        masses = np.array([0.13957039, 2.80839160743, 2.99131])
        p4 = four_vector([0.5, 2.0, 4.0], [-0.7, 0.0, 0.8], [0.0, 1.1, -2.2], masses)
        np.testing.assert_allclose(invariant_mass(p4), masses, atol=1e-13)
        np.testing.assert_allclose(
            p4[:, 1] ** 2 + p4[:, 2] ** 2, np.array([0.5, 2, 4]) ** 2
        )

    def test_kstar_is_cm_momentum_and_lorentz_invariant(self):
        q = np.array([0.0, 0.01, 0.2, 0.6])
        first = np.column_stack((np.sqrt(2.99131**2 + q * q), q, q * 0, q * 0))
        second = np.column_stack((np.sqrt(0.13957039**2 + q * q), -q, q * 0, q * 0))
        np.testing.assert_allclose(kstar(first, second), q, atol=1e-8)
        for beta in [-0.8, 0.0, 0.4, 0.9]:
            with self.subTest(beta=beta):
                actual = kstar(boost_x(first, beta), boost_x(second, beta))
                np.testing.assert_allclose(actual, q, atol=1e-7)
                np.testing.assert_allclose(
                    actual,
                    kstar(boost_x(second, beta), boost_x(first, beta)),
                    atol=1e-12,
                )

    def test_spacelike_particle_does_not_produce_a_physical_kstar(self):
        first = np.array([[1.0, 2.0, 0.0, 0.0]])
        second = np.array([[3.0, -1.0, 0.0, 0.0]])
        self.assertTrue(np.isnan(kstar(first, second)[0]))

    def test_its_cluster_nibbles_exclude_empty_layers(self):
        mean, count = decode_its_cluster_sizes([0, 0x7654321, 0x0000042])
        np.testing.assert_array_equal(count, [0, 7, 2])
        np.testing.assert_allclose(mean, [0, 4, 3])

    def test_derive_ct_pointing_and_configurable_he_momentum(self):
        physics = copy.deepcopy(DEFAULTS["physics"])
        frame = pd.DataFrame(
            {
                "PtHe3": [2.0],
                "EtaHe3": [0.0],
                "PhiHe3": [0.0],
                "PtPi": [0.2],
                "EtaPi": [0.0],
                "PhiPi": [0.0],
                "XPrimVtx": [0.3],
                "YPrimVtx": [-0.2],
                "ZPrimVtx": [1.0],
                "XDecVtx": [4.7],
                "YDecVtx": [-0.2],
                "ZDecVtx": [1.0],
                "PairPiTrackPt": [0.3],
                "PairPiTrackEta": [0.2],
                "PairPiTrackPhi": [2.5],
            }
        )
        result = derive(frame, physics, pairs=True)
        self.assertAlmostEqual(result.pt_h3l.iloc[0], 2.2)
        length = np.sqrt(4.7**2 + 0.2**2 + 1.0**2)
        self.assertAlmostEqual(result.ct.iloc[0], length * physics["mass_h3l"] / 2.2)
        self.assertAlmostEqual(result.cos_pa.iloc[0], 4.7 / length)
        self.assertAlmostEqual(result.decay_radius.iloc[0], np.hypot(4.7, 0.2))
        changed_pv = frame.copy()
        changed_pv[["XPrimVtx", "YPrimVtx", "ZPrimVtx"]] = [20, -15, 7]
        np.testing.assert_array_equal(
            derive(changed_pv, physics)[["ct", "cos_pa", "decay_radius"]],
            result[["ct", "cos_pa", "decay_radius"]],
        )
        self.assertTrue(np.isfinite(result.kstar.iloc[0]))
        self.assertNotIn("mass_h3l", frame)
        physics["he3_momentum_scale"] = 2
        self.assertAlmostEqual(derive(frame, physics).pt_h3l.iloc[0], 4.2)

    def test_zero_momentum_geometry_is_invalid_not_fabricated(self):
        frame = pd.DataFrame(
            {
                "PtHe3": [0.0],
                "EtaHe3": [0.0],
                "PhiHe3": [0.0],
                "PtPi": [0.0],
                "EtaPi": [0.0],
                "PhiPi": [0.0],
                "XPrimVtx": [0.0],
                "YPrimVtx": [0.0],
                "ZPrimVtx": [0.0],
                "XDecVtx": [0.0],
                "YDecVtx": [0.0],
                "ZDecVtx": [0.0],
            }
        )
        result = derive(frame, DEFAULTS["physics"])
        self.assertTrue(np.isnan(result.ct.iloc[0]))
        self.assertTrue(np.isnan(result.cos_pa.iloc[0]))


if __name__ == "__main__":
    unittest.main()
