import numpy as np
import pandas as pd

from ckm.core.kernels import IonisationKernels


class WindyStar(object):
    """ Windy star class.

    Model the projected line-of-sight orbital velocities of binary
    star systems for a variety of spectroscopic tracers in the
    time-domain.

    Methods
    -------

        configure_params : set the parameters of the windy stellar system.

        configure_epochs : set the epochs for returned data points.

        configure_kernel : set the kernel to be implemented in the
            convolutional model. Options: TopHat, RTInterp.

    Properties
    ----------

        keplerian_model_centroid_velocity : calculate velocity centroid of a
            binary keplerian orbit projected onto the line-of-sight.

        convolutional_model_centroid_velocity : calculate velocity centroid of
            convolved or time averaged binary Keplerian orbit projected onto
            the line-of-sight.

    """

    def __init__(self):
        # Public.
        self.time_of_periastron = None
        self.epochs = None
        self.volume = None
        self.period = None
        self.eccentricity = None
        self.semi_major_axis = None
        self.semi_minor_axis = None
        self.mass_a = None
        self.mass_b = None
        self.stellar_radius_a = None
        self.stellar_radius_b = None
        self.m_dot_a = None
        self.m_dot_b = None
        self.wind_velocity_a = None
        self.wind_velocity_b = None
        self.argument_of_periastron = None
        self.inclination = None
        self.rv_semi_amplitude = None
        self.ionisation_radius = None
        self.photospheric_radius = None
        self.log_sigma_time = None
        self.mean_emissivity = None
        self.effective_temperature_a = None
        self.systemic_velocity = None
        self.kernel_mode = None
        self.kernel_line = None
        self.interp_dims = None
        self.emissivity_interpolators = None

        # Private.
        self._tdi = None

    def __repr__(self):
        return 'Windy star object.'

    def configure_params(self, period=None, eccentricity=None,
                         semi_major_axis=None, mass_a=None, mass_b=None,
                         stellar_radius_a=None, stellar_radius_b=None,
                         m_dot_a=None, m_dot_b=None, wind_velocity_a=None,
                         wind_velocity_b=None, argument_of_periastron=None,
                         inclination=None, rv_semi_amplitude=None,
                         ionisation_radius=None, photospheric_radius=None,
                         log_sigma_time=None, mean_emissivity=None,
                         effective_temperature_a=None, rv_offset=0):
        """ Configure params. """
        # Primary star params.
        self.mass_a = mass_a
        self.stellar_radius_a = stellar_radius_a
        self.m_dot_a = m_dot_a
        self.wind_velocity_a = wind_velocity_a
        self.effective_temperature_a = effective_temperature_a

        # Secondary star params.
        self.mass_b = mass_b
        self.stellar_radius_b = stellar_radius_b
        self.m_dot_b = m_dot_b
        self.wind_velocity_b = wind_velocity_b

        # System params.
        self.period = period
        self.eccentricity = eccentricity
        if semi_major_axis == 'auto':
            self.semi_major_axis = ((self.period ** 2) *
                                    (self.mass_a + self.mass_b)) ** (1 / 3)
        else:
            self.semi_major_axis = semi_major_axis
        if semi_major_axis is not None:
            self.semi_minor_axis = self.semi_major_axis * np.sqrt(
                1 - pow(self.eccentricity, 2))
        if argument_of_periastron is not None:
            self.argument_of_periastron = argument_of_periastron * np.pi / 180
        if inclination is not None:
            self.inclination = inclination * np.pi / 180
        if rv_semi_amplitude == 'auto':
            self.rv_semi_amplitude = (29.78 / np.sqrt(1 - (self.eccentricity ** 2))) \
                * (self.mass_b * np.sin(self.inclination)) \
                * ((self.mass_b + self.mass_a) ** (-2 / 3)) \
                * ((self.period / 365) ** (-1 / 3))
        else:
            self.rv_semi_amplitude = rv_semi_amplitude
        self.ionisation_radius = ionisation_radius
        self.photospheric_radius = photospheric_radius
        self.log_sigma_time = log_sigma_time
        self.mean_emissivity = mean_emissivity
        self.systemic_velocity = rv_offset

    def configure_epochs(self, epochs, time_of_periastron=None):
        """ Configure epochs in phase units. """
        if time_of_periastron is not None:
            self.time_of_periastron = time_of_periastron
            self.epochs = ((epochs - self.time_of_periastron)
                           % self.period) / self.period
        else:
            self.epochs = epochs

    def configure_kernel(self, kernel_mode, kernel_line=None, interp_dims=None,
                         emissivity_interpolators=None):
        """ Configure kernel. """
        self.kernel_mode = kernel_mode
        self.kernel_line = kernel_line
        self.interp_dims = interp_dims
        self.emissivity_interpolators = emissivity_interpolators

    @property
    def keplerian_model_centroid_velocity(self):
        """ Calculate centroid velocity of a binary Keplerian orbit. """
        # Assert correct params have been set.
        try:
            assert self.period is not None
            assert self.eccentricity is not None
            assert self.rv_semi_amplitude is not None
            assert self.argument_of_periastron is not None
            assert self.systemic_velocity is not None
        except AssertionError as err:
            raise AssertionError(
                'Keplerian model for centroid velocity requires params period, '
                'eccentricity, semi-amplitude, argument of periastron and '
                'systemic velocity to be set.')

        # Calc orbital anomalies: mean, eccentric, true.
        self._orbital_anomalies()

        # LOS projection of Keplerian velocity.
        self._tdi['KeplerVelocity'] = self.rv_semi_amplitude * (
                (np.cos(self.argument_of_periastron + self._tdi['TrueAnomaly']))
                + (self.eccentricity * np.cos(self.argument_of_periastron)))

        # Systemic velocity constant.
        self._tdi['KeplerVelocity'] += self.systemic_velocity

        return self._tdi[['Phase', 'KeplerVelocity']]

    @property
    def convolutional_model_centroid_velocity(self):
        """ Calculate centroid velocity of convolved/time averaged Keplerian orbit. """
        # Assert correct params have been set.
        try:
            assert self.period is not None
            assert self.eccentricity is not None
            assert self.rv_semi_amplitude is not None
            assert self.argument_of_periastron is not None
            assert self.systemic_velocity is not None
        except AssertionError as err:
            raise AssertionError(
                'Convolutional model for centroid velocity requires params period, '
                'eccentricity, semi-amplitude, argument of periastron, systemic '
                'velocity to be set.')

        # Calc orbital anomalies: mean, eccentric, true.
        self._orbital_anomalies()

        # LOS projection of Keplerian velocity.
        self._tdi['KeplerVelocity'] = self.rv_semi_amplitude * (
                (np.cos(self.argument_of_periastron + self._tdi['TrueAnomaly']))
                + (self.eccentricity * np.cos(self.argument_of_periastron)))

        try:
            # Build kernel.
            ionisation_kernels = IonisationKernels(
                period=self.period, phase=self._tdi['Phase'])

            if self.kernel_mode == 'TopHat':
                kernel = ionisation_kernels.top_hat(
                    ionisation_radius=self.ionisation_radius,
                    photospheric_radius=self.photospheric_radius,
                    stellar_radius=self.stellar_radius_a,
                    wind_velocity_a=self.wind_velocity_a)

            elif self.kernel_mode == 'LogGaussian':
                kernel = ionisation_kernels.log_gaussian(
                    log_sigma_time=self.log_sigma_time,
                    mean_emissivity=self.mean_emissivity,
                    stellar_radius=self.stellar_radius_a,
                    wind_velocity_a=self.wind_velocity_a)

            elif self.kernel_mode == 'RTInterp':
                # Interpolation dimensions.
                ionisation_kernels.emissivity_obj = self.emissivity_interpolators
                if self.interp_dims == 'Point':
                    kernel = ionisation_kernels.static_radiative_transfer(
                        line=self.kernel_line)
                elif self.interp_dims == 'Line':
                    kernel = ionisation_kernels.radiative_transfer_interpolation_1d(
                        m_dot_a=self.m_dot_a,
                        line=self.kernel_line)
                elif self.interp_dims == 'Grid':
                    kernel = ionisation_kernels.radiative_transfer_interpolation_2d(
                        m_dot_a=self.m_dot_a,
                        wind_velocity_a=self.wind_velocity_a,
                        line=self.kernel_line)
                else:
                    raise NameError('Interp dims must be set. Options '
                                    'available: Point(0D), Line(1D), Grid(2D).')

            else:
                raise NameError('Kernel type must be set. Options '
                                'available: TopHat, RTInterp.')

            # Perform convolution/moving average. Mode=valid ensures convolution
            # is only computed where signals have complete overlap. Convolution is
            # mathematically defined to flip kernel, and so backwards in time is
            # increasing radial duration as required.
            moving_average = np.convolve(
                self._tdi['KeplerVelocity'].values, kernel, mode='valid')

            # Pre-pend nan to fill moving average array to the same length.
            # By pre-pending we line up the convolved velocity array with the
            # mean anomaly array such that data is an average of past motion.
            final_convolution = np.append(np.full(len(kernel) - 1, np.nan), moving_average)
            self._tdi['ConvolvedVelocity'] = pd.Series(final_convolution)

            # Systemic velocity constant.
            self._tdi['ConvolvedVelocity'] += self.systemic_velocity

            return self._tdi[['Phase', 'ConvolvedVelocity']]

        except ValueError as err:
            raise ValueError('Convolution requires higher time resolution, '
                             'or parameters have walked outside interpolation range.')

    def _orbital_anomalies(self):
        """ Orbital anomalies: mean, eccentric, true. """
        # Phase.
        self._tdi = pd.DataFrame(self.epochs.reshape(
            len(self.epochs), 1), columns=['Phase'])

        # Mean anomaly.
        self._tdi['MeanAnomaly'] = self._tdi['Phase'] * (2 * np.pi)

        # Eccentric anomaly.
        try:
            self._tdi['EccentricAnomaly'] = \
                self._tdi['MeanAnomaly'].apply(
                    lambda phi: self._newtonian_raphson_method_for_keplers_equation(
                        E_i=phi, M=phi, e=self.eccentricity))
        except RecursionError as err:
            print('Recursion error caught, nbd.')
            self._tdi['EccentricAnomaly'] = np.nan

        # True anomaly.
        numerator = np.sqrt(1 + self.eccentricity) \
                * np.sin(self._tdi['EccentricAnomaly'] / 2)
        denominator = np.sqrt(1 - self.eccentricity) \
                * np.cos(self._tdi['EccentricAnomaly'] / 2)
        self._tdi['TrueAnomaly'] = 2 * np.arctan2(numerator, denominator)

    def _newtonian_raphson_method_for_keplers_equation(self, E_i, M, e):
        """ Numerical solution to Kepler's equation. """
        if np.isnan(M):
            # Check if nan fed in.
            return np.nan

        # Iteration step.
        E_i_1 = (M + (e * (np.sin(E_i) - (E_i * np.cos(E_i)))))\
                / (1 - (e * np.cos(E_i)))

        # Convergence test.
        epsilon = 1e-10
        if not abs(E_i_1 - E_i) < epsilon:
            return self._newtonian_raphson_method_for_keplers_equation(E_i_1, M, e)

        # Solution converged.
        return E_i_1


windy_star = WindyStar()
