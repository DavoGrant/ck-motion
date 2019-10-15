import numpy as np

from ckm.core.utils import convert_km_per_s_to_au_per_day, \
    convert_time_to_phase_units


class IonisationKernels(object):
    """ Ionisation kernels class.

    Ionisation kernels of normalised emissivity in time prepared
    for convolving with a mean anomaly array of evenly spaced indices.

    Methods
    -------

        top_hat : simple flat kernel assuming all radii between the
            optically thick boundary and an outer ionisation radius
            contribute equal line emission as per conservation of
            mass. Wind assumed to be at terminal velocity at all
            radii.

        log_gaussian : Gaussian in logarithmic space, otherwise same
            as for top_hat kernel.

        static_radiative_transfer : kernel is a curve computed
            from a radiative transfer result.

        radiative_transfer_interpolation_1d : kernel is a curve computed
            by interpolating a line of radiative transfer results.

        radiative_transfer_interpolation_2d : kernel is a curve computed
            by interpolating a grid of radiative transfer results.

    """

    def __init__(self, period, phase):
        self.period = period
        self.phase = phase
        self.emissivity_obj = None

    def __repr__(self):
        return 'Ionisation kernels object.'

    def top_hat(self, ionisation_radius, photospheric_radius,
                stellar_radius, wind_velocity_a):
        """ Top hat function. """
        # Assert correct params have been set.
        try:
            assert ionisation_radius is not None
            assert photospheric_radius is not None
            assert stellar_radius is not None
            assert wind_velocity_a is not None
        except AssertionError as err:
            raise AssertionError(
                'Top-hat kernel requires params ionisation radius, '
                'photospheric radius, stellar radius, terminal wind '
                'velocity to be set.')

        # Travel time from photospheric radius to ionisation radius in phase units.
        # Assuming wind terminal velocity.
        tt_phot_to_ion_r = convert_time_to_phase_units(
            (ionisation_radius - photospheric_radius)
            / convert_km_per_s_to_au_per_day(wind_velocity_a), self.period)

        # Travel time from stellar radius to photospheric radius in phase units.
        # Assuming wind terminal velocity.
        tt_star_to_phot_r = convert_time_to_phase_units(
            (photospheric_radius - stellar_radius)
            / convert_km_per_s_to_au_per_day(wind_velocity_a), self.period)

        # Mean anomaly per unit array index.
        mean_anomaly = self.phase * (2 * np.pi)
        upi = (mean_anomaly.max() - mean_anomaly.min()) / len(mean_anomaly)

        # Number of array indices for phot to ion radius travel.
        tt_phot_to_ion_indices = tt_phot_to_ion_r / upi

        # Number of array indices for star to phot radius travel.
        tt_star_to_phot_indices = tt_star_to_phot_r / upi

        # Optically thick/zero emissivity appended after top hat
        # as backwards in time is increasing radial duration.
        phot_kernel = np.zeros(int(tt_star_to_phot_indices))
        emission_vol_kernel = np.ones(int(tt_phot_to_ion_indices))
        kernel = np.append(phot_kernel, emission_vol_kernel)

        # Build normalised kernel.
        norm_kernel = kernel / np.linalg.norm(kernel, ord=1)

        return norm_kernel

    def log_gaussian(self, log_sigma_time, mean_emissivity,
                     stellar_radius, wind_velocity_a):
        """ Logarithmic Gaussian function. """
        # Assert correct params have been set.
        try:
            assert log_sigma_time is not None
            assert mean_emissivity is not None
            assert stellar_radius is not None
            assert wind_velocity_a is not None
        except AssertionError as err:
            raise AssertionError(
                'Log Gaussian kernel requires params log sigma time, emissivity '
                'mean, terminal wind velocity and spectral line to be set.')

        # Consider radii up to 200 AU. Limiting this ensure complete
        # overlap in the convolution's valid mode.
        log_radii = np.linspace(np.log10(stellar_radius), 2.3, len(self.phase))

        # Transform Gaussian mean into logarithmic space.
        log_mean_emissivity = np.log10(mean_emissivity)

        # Gaussian in logarithmic space.
        emissivities = np.exp(-(log_radii - log_mean_emissivity) ** 2
                              / (2. * log_sigma_time ** 2))

        # Convert radii to time in days. Assuming wind terminal velocity.
        radii = np.power(10, log_radii)
        times = (radii - stellar_radius) \
                / convert_km_per_s_to_au_per_day(wind_velocity_a)

        # Interpolate in 1d to get the emissivity at compatible per index phase
        # intervals in the kernel, required for convolution.
        _Time = self.phase * self.period
        _Time = _Time[_Time < times.max()]
        kernel = np.interp(_Time, times, emissivities)

        # Build normalised kernel.
        kernel_nm = kernel / np.linalg.norm(kernel, ord=1)

        return kernel_nm

    def static_radiative_transfer(self, line):
        """ Radiative transfer static curve. """
        # Assert correct params have been set.
        try:
            assert line is not None
        except AssertionError as err:
            raise AssertionError(
                'Radiative transfer static kernel requires params '
                'for spectral line to be set.')

        # Prepare slice of epochs to interpolate along. By using original
        # phase series as template we ensure compatible per index phase
        # intervals in the kernel, required for convolution.
        epochs = np.zeros((len(self.phase), 1))
        epochs[:, 0] = self.phase * self.period

        # Select spectral line.
        self.emissivity_obj.set_spectral_line(line=line)
        kernel = self.emissivity_obj.emissivity_interpolation_fn(epochs)

        # Remove indices outside interp range which have been set to
        # nan in the RegularGridInterpolator object.
        kernel_rm = kernel[~np.isnan(kernel)]

        # Build normalised kernel.
        kernel_nm = kernel_rm / np.linalg.norm(kernel_rm, ord=1)

        # import matplotlib.pyplot as plt
        # plt.plot(self.phase[:len(kernel_nm)] * self.period, kernel_nm)
        # plt.show()

        return kernel_nm

    def radiative_transfer_interpolation_1d(self, m_dot_a, line):
        """ 1D radiative transfer interpolated curve. """
        # Assert correct params have been set.
        try:
            assert m_dot_a is not None
            assert line is not None
        except AssertionError as err:
            raise AssertionError(
                'Radiative transfer interpolation kernel requires params '
                'mass loss rate and spectral line to be set.')

        # Prepare slice of epochs to interpolate along. By using original
        # phase series as template we ensure compatible per index phase
        # intervals in the kernel, required for convolution.
        epochs = np.zeros((len(self.phase), 2))
        epochs[:, 0] = m_dot_a
        epochs[:, 1] = self.phase * self.period

        # Interpolate spectral line.
        self.emissivity_obj.set_spectral_line(line=line)
        kernel = self.emissivity_obj.emissivity_interpolation_fn(epochs)

        # Remove indices outside interp range which have been set to
        # nan in the RegularGridInterpolator object.
        kernel_rm = kernel[~np.isnan(kernel)]

        # Build normalised kernel.
        kernel_nm = kernel_rm / np.linalg.norm(kernel_rm, ord=1)

        # import matplotlib.pyplot as plt
        # plt.plot(self.phase[:len(kernel_nm)] * self.period, kernel_nm)
        # plt.show()

        return kernel_nm

    def radiative_transfer_interpolation_2d(self, m_dot_a, wind_velocity_a,
                                            line):
        """ 2D radiative transfer interpolated curve. """
        # Assert correct params have been set.
        try:
            assert m_dot_a is not None
            assert wind_velocity_a is not None
            assert line is not None
        except AssertionError as err:
            raise AssertionError(
                'Radiative transfer interpolation kernel requires params '
                'mass loss rate, terminal wind velocity and spectral '
                'line to be set.')

        # Prepare slice of epochs to interpolate along. By using original
        # phase series as template we ensure compatible per index phase
        # intervals in the kernel, required for convolution.
        epochs = np.zeros((len(self.phase), 3))
        epochs[:, 0] = m_dot_a
        epochs[:, 1] = wind_velocity_a
        epochs[:, 2] = self.phase * self.period

        # Interpolate spectral line.
        self.emissivity_obj.set_spectral_line(line=line)
        kernel = self.emissivity_obj.emissivity_interpolation_fn(epochs)

        # Remove indices outside interp range which have been set to
        # nan in the RegularGridInterpolator object.
        kernel_rm = kernel[~np.isnan(kernel)]

        # Build normalised kernel.
        kernel_nm = kernel_rm / np.linalg.norm(kernel_rm, ord=1)

        # import matplotlib.pyplot as plt
        # plt.plot(self.phase[:len(kernel_nm)] * self.period, kernel_nm)
        # plt.show()

        return kernel_nm
