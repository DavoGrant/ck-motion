import unittest
import numpy as np
import pandas as pd

from ckm.core.motion import windy_star
from ckm.optimise.mcmc_kepler import MCMCOptimiser


class TestKeplerianModelling(unittest.TestCase):
    """ Test Keplerian modelling. """

    def __init__(self, *args, **kwargs):
        super(TestKeplerianModelling, self).__init__(*args, **kwargs)

    def test_keplerian_circular_orbit_model(self):
        """ Test Keplerian circular orbit model. """
        # Arbitrary circular orbit.
        windy_star.configure_params(
            period=1000, eccentricity=0, rv_semi_amplitude=50,
            argument_of_periastron=270, rv_offset=0)
        windy_star.configure_epochs(
            epochs=np.linspace(0, 2, 1000))
        res_df = windy_star.keplerian_model_centroid_velocity

        # Sanity checks.
        self.assertAlmostEqual(
            res_df['KeplerVelocity'].max(), 50, places=0,
            msg='Keplerian velocity unexpected max value.')
        self.assertAlmostEqual(
            res_df['KeplerVelocity'].min(), -50, places=0,
            msg='Keplerian velocity unexpected min value.')
        self.assertAlmostEqual(
            res_df['KeplerVelocity'].iloc[0], 0, places=0,
            msg='Keplerian velocity unexpected value at phase 0.')
        self.assertAlmostEqual(
            res_df['KeplerVelocity'].iloc[124], 50, places=0,
            msg='Keplerian velocity unexpected value at phase 0.25')
        self.assertAlmostEqual(
            res_df['KeplerVelocity'].iloc[249], 0, places=0,
            msg='Keplerian velocity unexpected value at phase 0.5')
        self.assertAlmostEqual(
            res_df['KeplerVelocity'].iloc[374], -50, places=0,
            msg='Keplerian velocity unexpected value at phase 0.75')

    def test_keplerian_eccentric_orbit_model(self):
        """ Test Keplerian eccentric orbit model. """
        # Arbitrary eccentric orbit.
        windy_star.configure_params(
            period=1000, eccentricity=0.9, rv_semi_amplitude=50,
            argument_of_periastron=270, rv_offset=0)
        windy_star.configure_epochs(
            epochs=np.linspace(0, 2, 1000))
        res_df = windy_star.keplerian_model_centroid_velocity

        # Sanity checks.
        self.assertAlmostEqual(
            res_df['KeplerVelocity'].max(), 50, places=0,
            msg='Keplerian velocity unexpected max value.')
        self.assertAlmostEqual(
            res_df['KeplerVelocity'].min(), -50, places=0,
            msg='Keplerian velocity unexpected min value.')
        self.assertAlmostEqual(
            res_df['KeplerVelocity'].iloc[0], 0, places=0,
            msg='Keplerian velocity unexpected value at phase 0.')
        self.assertNotAlmostEqual(
            res_df['KeplerVelocity'].iloc[124], 50, places=0,
            msg='Keplerian velocity unexpected value at phase 0.25')
        self.assertAlmostEqual(
            res_df['KeplerVelocity'].iloc[249], 0, places=0,
            msg='Keplerian velocity unexpected value at phase 0.5')
        self.assertNotAlmostEqual(
            res_df['KeplerVelocity'].iloc[374], -50, places=0,
            msg='Keplerian velocity unexpected value at phase 0.75')

    def test_keplerian_mcmc_optimisation(self):
        """ Test Keplerian MCMC optimisation. """
        # Synthetic observations.
        obs = pd.DataFrame()
        obs['JD'] = np.linspace(2450000, 2451000, 1000)
        obs['Velocity'] = 100 * np.sin((obs['JD'] - 2450000) / 100)
        obs['VelocityError'] = np.ones(1000)

        # Optimise for 12 walkers and one chain step.
        op = MCMCOptimiser()
        op.optimise_kepler_model(
            observations=obs,
            fixed_params={'Period': 2022.7},
            guess=np.array([245000, 0.1, 98, 265, 0]),
            priors={'TimeOfPeriastron': (244990, 245010),
                    'Eccentricity': (1e-3, 0.5),
                    'RVSemiAmplitude': (50, 150),
                    'ArgumentOfPeriastron': (180, 360),
                    'RVOffset': (-50, 50)},
            walkers=12, burn=2, run=2)


if __name__ == '__main__':
    unittest.main()
