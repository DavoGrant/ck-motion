import unittest
import numpy as np

from ckm.radiative.emissivity import Emissivity
from ckm.core.motion import windy_star


class TestConvolutionalModelling(unittest.TestCase):
    """ Test Keplerian modelling. """

    def __init__(self, *args, **kwargs):
        super(TestConvolutionalModelling, self).__init__(*args, **kwargs)

    def test_convolutional_model_top_hat_kernel_circular(self):
        """ Test ck-motion w/ top hat kernel for circular orbit. """
        # Arbitrary circular orbit.
        windy_star.configure_params(
            period=1000, eccentricity=0, rv_semi_amplitude=50,
            argument_of_periastron=270, rv_offset=0,
            ionisation_radius=50, photospheric_radius=5,
            stellar_radius_a=1, wind_velocity_a=500)
        windy_star.configure_epochs(
            epochs=np.linspace(0, 2, 1000))
        windy_star.configure_kernel(
            kernel_mode='TopHat')
        res_df = windy_star.convolutional_model_centroid_velocity

        # Sanity checks.
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].max(), 50, places=0,
            msg='Convolutional velocity unexpected max value.')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].min(), -50, places=0,
            msg='Convolutional velocity unexpected min value.')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[0], 0, places=0,
            msg='Convolutional velocity unexpected value at phase 0.')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[124], 50, places=0,
            msg='Convolutional velocity unexpected value at phase 0.25')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[249], 0, places=0,
            msg='Convolutional velocity unexpected value at phase 0.5')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[374], -50, places=0,
            msg='Convolutional velocity unexpected value at phase 0.75')

    def test_convolutional_model_top_hat_kernel_eccentric(self):
        """ Test ck-motion w/ top hat kernel for eccentric orbit. """
        # Arbitrary circular orbit.
        windy_star.configure_params(
            period=1000, eccentricity=0.9, rv_semi_amplitude=50,
            argument_of_periastron=270, rv_offset=0,
            ionisation_radius=50, photospheric_radius=5,
            stellar_radius_a=1, wind_velocity_a=500)
        windy_star.configure_epochs(
            epochs=np.linspace(0, 2, 1000))
        windy_star.configure_kernel(
            kernel_mode='TopHat')
        res_df = windy_star.convolutional_model_centroid_velocity

        # Sanity checks.
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].max(), 50, places=0,
            msg='Convolutional velocity unexpected max value.')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].min(), -50, places=0,
            msg='Convolutional velocity unexpected min value.')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[0], 0, places=0,
            msg='Convolutional velocity unexpected value at phase 0.')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[124], 50, places=0,
            msg='Convolutional velocity unexpected value at phase 0.25')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[249], 0, places=0,
            msg='Convolutional velocity unexpected value at phase 0.5')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[374], -50, places=0,
            msg='Convolutional velocity unexpected value at phase 0.75')

    def test_convolutional_model_log_gaussain_kernel_circular(self):
        """ Test ck-motion w/ log gaussian kernel for circular orbit. """
        # Arbitrary circular orbit.
        windy_star.configure_params(
            period=1000, eccentricity=0, rv_semi_amplitude=50,
            argument_of_periastron=270, rv_offset=0,
            log_sigma_time=0.5, mean_emissivity=30,
            stellar_radius_a=1, wind_velocity_a=500)
        windy_star.configure_epochs(
            epochs=np.linspace(0, 2, 1000))
        windy_star.configure_kernel(
            kernel_mode='LogGaussian')
        res_df = windy_star.convolutional_model_centroid_velocity

        # Sanity checks.
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].max(), 50, places=0,
            msg='Convolutional velocity unexpected max value.')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].min(), -50, places=0,
            msg='Convolutional velocity unexpected min value.')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[0], 0, places=0,
            msg='Convolutional velocity unexpected value at phase 0.')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[124], 50, places=0,
            msg='Convolutional velocity unexpected value at phase 0.25')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[249], 0, places=0,
            msg='Convolutional velocity unexpected value at phase 0.5')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[374], -50, places=0,
            msg='Convolutional velocity unexpected value at phase 0.75')

    def test_convolutional_model_log_gaussain_kernel_eccentric(self):
        """ Test ck-motion w/ log gaussian kernel for eccentric orbit. """
        # Arbitrary circular orbit.
        windy_star.configure_params(
            period=1000, eccentricity=0.9, rv_semi_amplitude=50,
            argument_of_periastron=270, rv_offset=0,
            log_sigma_time=0.3, mean_emissivity=30,
            stellar_radius_a=1, wind_velocity_a=500)
        windy_star.configure_epochs(
            epochs=np.linspace(0, 2, 1000))
        windy_star.configure_kernel(
            kernel_mode='LogGaussian')
        res_df = windy_star.convolutional_model_centroid_velocity

        # Sanity checks.
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].max(), 50, places=0,
            msg='Convolutional velocity unexpected max value.')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].min(), -50, places=0,
            msg='Convolutional velocity unexpected min value.')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[0], 0, places=0,
            msg='Convolutional velocity unexpected value at phase 0.')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[124], 50, places=0,
            msg='Convolutional velocity unexpected value at phase 0.25')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[249], 0, places=0,
            msg='Convolutional velocity unexpected value at phase 0.5')
        self.assertNotAlmostEqual(
            res_df['ConvolvedVelocity'].iloc[374], -50, places=0,
            msg='Convolutional velocity unexpected value at phase 0.75')

    def test_make_interpolation(self):
        """ Test Keplerian circular orbit model. """
        # Instantiate emissivity object in setup mode.
        emissivity_setup = Emissivity(setup_mode=True)
        emissivity_setup.rt_results_db = 'test_jar/mock_rt_results.db'
        emissivity_setup.rt_results_table = 'mock_data'

        # Setup spectral line and wind velocity profile.
        emissivity_setup.set_spectral_line(line=3)
        emissivity_setup.set_wind_velocity_profile(
            wind_velocity_solution='beta_1',
            initial_velocity=20,
            stellar_radius=50,
            sonic_radius=60)

        # Build interpolation object.
        Mdot = 5e-4
        Vinf = 500
        Time = np.linspace(0, 1000, 10)
        emissivity_setup.build_static_interpolation_object(Mdot, Vinf, Time)

    def test_convolutional_model_rt_interp_kernel_circular(self):
        """ Test ck-motion w/ RT interp kernel for circular orbit. """
        # Arbitrary circular orbit.
        emissivity = Emissivity(setup_mode=False, pickle_dir='test_jar')
        windy_star.configure_params(
            period=2022.7, eccentricity=0, rv_semi_amplitude=100,
            argument_of_periastron=270, rv_offset=0,
            m_dot_a=5e-4, wind_velocity_a=500)
        windy_star.configure_epochs(epochs=np.linspace(0, 2, 1000))
        windy_star.configure_kernel(kernel_mode='RTInterp',
                                    kernel_line=3, interp_dims='Grid',
                                    emissivity_interpolators=emissivity)
        res_df = windy_star.convolutional_model_centroid_velocity

    def test_convolutional_model_rt_interp_kernel_eccentric(self):
        """ Test ck-motion w/ RT interp kernel for eccentric orbit. """
        # Arbitrary eccentric orbit.
        emissivity = Emissivity(setup_mode=False, pickle_dir='test_jar')
        windy_star.configure_params(
            period=2022.7, eccentricity=0.9, rv_semi_amplitude=100,
            argument_of_periastron=270, rv_offset=0,
            m_dot_a=5e-4, wind_velocity_a=500)
        windy_star.configure_epochs(epochs=np.linspace(0, 2, 1000))
        windy_star.configure_kernel(kernel_mode='RTInterp',
                                    kernel_line=3, interp_dims='Grid',
                                    emissivity_interpolators=emissivity)
        res_df = windy_star.convolutional_model_centroid_velocity


if __name__ == '__main__':
    unittest.main()
