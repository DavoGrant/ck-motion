import os
import pickle
import sqlite3
import datetime
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pylab as plt


class Emissivity(object):
    """ Emissivity class.

    Emissivity of various spectral lines as a function of wind travel
    time as calculated by a suitable radiative transfer code.
    Emissivity-time curves are formed by n-d interpolation of a grid
    of results for varying parameterisations of the star and its wind.

    Methods
    -------

        set_spectral_line : Set spectral line transition.

        build_static_interpolation_object : Build static point
            interpolation object.

        build_1d_interpolation_object : Build 1d line interpolation
            object.

        build_2d_interpolation_object : Build 2d grid interpolation
            object.

        pickle_interpolation_object : Write grid interpolation object
            to disk.

    Properties
    ----------

        emissivity_interpolation_fn : scipy.interpolate.RegularGridInterpolator
            object, returned from a dictionary of options for each spectral
            line. Line must be pre-set by method set_spectral_line().

    """

    def __init__(self, setup_mode=False, pickle_dir=None):
        self.draw = False
        self.setup_mode = setup_mode
        self.pickle_dir = pickle_dir
        self.rt_results_db = None
        self.rt_results_table = None
        self.line = None
        self.wind_velocity_solution = None
        self.initial_velocity = None
        self.stellar_radius = None
        self.sonic_radius = None
        self.g_smooth_sigma = 1
        self._grid_interp_fn = None
        self._epsilon = None
        self._grid_interp_collection = {}

        if not self.setup_mode:
            # Load interpolation object at instantiation, once only.
            self._unpickle_emissivity_interpolation_objects()

    def __repr__(self):
        return 'Emissivity object. Setup mode={}'.format(self.setup_mode)

    def set_spectral_line(self, line):
        """ Set spectral line transition. """
        self.line = line

    def set_wind_velocity_profile(self, wind_velocity_solution, initial_velocity,
                                  stellar_radius, sonic_radius):
        """ Set spectral line transition. """
        self.wind_velocity_solution = wind_velocity_solution  # km/s.
        self.initial_velocity = initial_velocity  # km/s.
        self.stellar_radius = stellar_radius * 6.957e10  # Solar radii to cm.
        self.sonic_radius = sonic_radius * 6.957e10  # Solar radii to cm.

    def _unpickle_emissivity_interpolation_objects(self):
        """ Load emissivity interpolation objects for all lines. """
        print('Loading interpolation collection.')
        # Scan jar for pickles.
        for root, dirs, files in os.walk(self.pickle_dir):
            for file in files:
                if file.endswith('.p'):

                    # Unpack interp obj to dict.
                    pickle_dict = pickle.load(open(
                        os.path.join(root, file), 'rb'))
                    line = pickle_dict['line']
                    dt = pickle_dict['datetime']
                    interp_fn = pickle_dict['interpolation_obj']
                    self._grid_interp_collection[line] = interp_fn

    @property
    def emissivity_interpolation_fn(self):
        """ Emissivity interpolation object. """
        # Assert spectral line set.
        self._ensure_setup_requirements()

        # Select interpolation fn for line.
        try:
            self._grid_interp_fn = self._grid_interp_collection[self.line]
        except KeyError as err:
            raise KeyError('Interpolation object not found for input kernel '
                           'line. Check the interpolation objects available '
                           'in the interp_jar or build a new one.')

        return self._grid_interp_fn

    def build_static_interpolation_object(self, Mdot, Vinf, Time):
        """ Build static point interpolation object. """
        # Assert spectral line set.
        self._ensure_setup_requirements()

        # Grid parameterisations: must be evenly spaced.
        # Emissivity at each part of parameter space.
        self._epsilon = np.zeros(len(Time))
        for i in range(len(Time)):
            self._epsilon[i] = self._lookup_rt_emissivity(
                Mdot, Vinf, Time[i])

        # Build interpolation object. Set beyond interp range to return nan.
        # N.B time bounds provides a kernel size which ensures complete
        # overlap in the convolution's valid mode for a given
        # number/fraction of signal/periods.
        self._grid_interp_fn = interpolate.RegularGridInterpolator(
            (Time,), self._epsilon,
            bounds_error=False, fill_value=np.nan)

    def build_1d_interpolation_object(self, Mdot, Vinf, Time):
        """ Build 1d line interpolation object. """
        # Assert spectral line set.
        self._ensure_setup_requirements()

        # Grid parameterisations: must be evenly spaced.
        # Emissivity at each part of parameter space.
        self._epsilon = np.zeros((len(Mdot), len(Time)))
        for i in range(len(Mdot)):
            for j in range(len(Time)):
                self._epsilon[i, j] = self._lookup_rt_emissivity(
                    Mdot[i], Vinf, Time[j])

        # Build interpolation object. Set beyond interp range to return nan.
        # N.B time bounds provides a kernel size which ensures complete
        # overlap in the convolution's valid mode for a given
        # number/fraction of signal/periods.
        self._grid_interp_fn = interpolate.RegularGridInterpolator(
            (Mdot, Time), self._epsilon,
            bounds_error=False, fill_value=np.nan)

    def build_2d_interpolation_object(self, Mdot, Vinf, Time):
        """ Build 2d grid interpolation object. """
        # Assert spectral line set.
        self._ensure_setup_requirements()

        # Grid parameterisations: must be evenly spaced.
        # Emissivity at each part of parameter space.
        self._epsilon = np.zeros((len(Mdot), len(Vinf), len(Time)))
        for i in range(len(Mdot)):
            for j in range(len(Vinf)):
                for k in range(len(Time)):
                    self._epsilon[i, j, k] = self._lookup_rt_emissivity(
                        Mdot[i], Vinf[j], Time[k])

        # Build interpolation object. Set beyond interp range to return nan.
        # N.B time bounds provides a kernel size which ensures complete
        # overlap in the convolution's valid mode for a given
        # number/fraction of signal/periods.
        self._grid_interp_fn = interpolate.RegularGridInterpolator(
            (Mdot, Vinf, Time), self._epsilon,
            bounds_error=False, fill_value=np.nan)

    def _ensure_setup_requirements(self):
        """ Throw errors if incomplete setup. """
        try:
            assert self.line is not None
        except AssertionError as err:
            raise AssertionError(
                'Spectral line must first be set. '
                'Call method set_spectral_line()')

        if self.setup_mode:
            try:
                assert self.wind_velocity_solution is not None
                if self.wind_velocity_solution == 'constant':
                    assert self.sonic_radius is not None
                elif self.wind_velocity_solution == 'beta_1':
                    assert self.initial_velocity is not None
                    assert self.stellar_radius is not None
                    assert self.sonic_radius is not None
            except AssertionError as err:
                raise AssertionError(
                    'Wind velocity profile incomplete setup. '
                    'Call method set_wind_velocity_profile()')

    def _lookup_rt_emissivity(self, _Mdot, _Vinf, _Time):
        """ Lookup radiative transfer emissivity results. """
        # Read rt results from DB table.
        query = 'SELECT * FROM {} '.format(self.rt_results_table)
        connection = sqlite3.connect(self.rt_results_db)
        data = pd.read_sql_query(query, connection)
        connection.close()

        # Convert types.
        data['Stellar_wind.mdot(msol/yr)'] = data[
            'Stellar_wind.mdot(msol/yr)'].astype('float')
        data['Stellar_wind.v_infinity(cm)'] = data[
            'Stellar_wind.v_infinity(cm)'].astype('float')
        data['line'] = data['line'].astype('int')
        data['radius_cm'] = data['radius_cm'].astype('float')
        data['linear_luminosity_density'] = data[
            'linear_luminosity_density'].astype('float')

        # Select rt run results.
        data['Stellar_wind.v_infinity(km)'] = round(data['Stellar_wind.v_infinity(cm)'] * 1e-5, 2)
        data_run = data.loc[(data['Stellar_wind.mdot(msol/yr)'] == _Mdot)
                            & (data['Stellar_wind.v_infinity(km)'] == _Vinf)].copy()

        # Select line by line id - see line_label for verbose transition name.
        data_line = data_run.loc[data_run['line'] == self.line].copy()

        # Pre-sort by radius for interpolation func.
        data_line = data_line.sort_values(
            by='radius_cm', ascending=True).reset_index(drop=True)

        # Extract emissivities and radii in cm.
        radii = data_line['radius_cm']
        emissivities = data_line['linear_luminosity_density']

        # Convert from radius to time.
        times = self._convert_radius_to_time(radii, _Vinf)

        # Stack all negative times, from inside co-orbiting radius, as
        # weighting for zero time velocity.
        co_orbiting_emissivities = emissivities[times <= 0]
        co_orbiting_total_emissivity = co_orbiting_emissivities.sum()
        emissivities = emissivities.loc[times > 0]
        times = times.loc[times > 0]
        emissivities = np.hstack((co_orbiting_total_emissivity, emissivities))
        times = np.hstack((0, times))

        if not np.all(np.diff(times) > 0):
            raise IndexError('Emissivity curve 1d interpolation requires x vals '
                             'are in ascending order, otherwise results are nonsense.')

        # Interpolate in 1d to get the emissivity at the time requested.
        # Times are chosen to be evenly sampled so won't match data points after
        # conversion, hence need for interpolation.
        # Gaussian smooth is first applied to negate noise effecting lin interp.
        # N.B default behavior is constant extrapolation.
        emissivities = gaussian_filter(emissivities, sigma=self.g_smooth_sigma)
        _emissivity = np.interp(_Time, times, emissivities)

        if self.draw:
            # Inspect time domain emissivity interpolation.
            plt.plot(times, emissivities)
            plt.scatter(_Time, _emissivity)
            # plt.xscale('log')
            plt.show()

        return _emissivity

    def _convert_radius_to_time(self, _radii, terminal_velocity):
        """ Convert emissivity curves from radius (cm) into time (days). """
        if self.wind_velocity_solution == 'constant':
            # Constant solution.
            _times = (_radii - self.sonic_radius) / terminal_velocity
            _times *= 1e-5 / (3600 * 24)

        elif self.wind_velocity_solution == 'beta_1':
            # Integrated beta velocity law: beta=1.
            _times = ((self.stellar_radius * (terminal_velocity - self.initial_velocity)
                       * np.log((self.initial_velocity * self.stellar_radius)
                                - (terminal_velocity * self.stellar_radius)
                                + (terminal_velocity * _radii)))
                      + (terminal_velocity * _radii)) / (terminal_velocity ** 2)
            _times -= ((self.stellar_radius * (terminal_velocity - self.initial_velocity)
                        * np.log((self.initial_velocity * self.stellar_radius)
                                 - (terminal_velocity * self.stellar_radius)
                                 + (terminal_velocity * self.sonic_radius)))
                       + (terminal_velocity * self.sonic_radius)) / (terminal_velocity ** 2)
            _times *= 1e-5 / (3600 * 24)

        else:
            raise NameError('Radius to duration conversion solution not '
                            'recognised. Options available: constant, beta_1.')

        # if len(_times[_times < 0]) > 0:
        #     raise ValueError('Negative times from inside co-orbiting radius. '
        #                      'Problems w/ sonic point or rt sim config?')

        return _times

    def pickle_interpolation_object(self, label):
        """ Write grid interpolation object to disk. """
        p_name = 'interp_object_line{}.p'.format(self.line)
        pickle_path = os.path.join('interp_jar', label, p_name)
        unique_time = str(datetime.datetime.utcnow()).replace(
            '-', '').replace(' ', '_').replace(':', '')[:15]

        # Ensure location exists.
        pickle_dir = os.path.join('interp_jar', label)
        if not os.path.exists(pickle_dir):
            os.mkdir(pickle_dir)

        # Pickle interpolation object in dict with meta data.
        print('Writing interpolation dict to disk.')
        pickle_dict = {
            'line': self.line,
            'datetime': unique_time,
            'interpolation_obj': self._grid_interp_fn}
        pickle.dump(pickle_dict, open(pickle_path, 'wb'))
