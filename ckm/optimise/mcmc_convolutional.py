import time
import emcee
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from ckm.core.motion import windy_star
from ckm.core.utils import calc_mcmc_chain_percentiles


class MCMCOptimiser(object):
    """ Markov Chain Monte Carlo class."""

    def __init__(self, kernel_type, interp_dims, emissivity_obj=None,
                 epoch_density=1000):
        self.debug = False
        self.observations = {}
        self.kernel_type = kernel_type
        self.interp_dims = interp_dims
        self.emissivity_obj = emissivity_obj
        self.epoch_density = epoch_density
        self._line_label = None
        self._line_data = None
        self._line_index = 1
        self.fixed_params = None
        self.priors = None
        self.initial_spread = 1e-3
        self.fig = None

    def __repr__(self):
        return 'Markov Chain Monte Carlo Optimiser Object for ' \
               'Convolutional Keplerian centroid velocity model.'

    def optimise_convolutional_model(self, observations, fixed_params, guess,
                                     priors, walkers=64, burn=500, run=500):
        """ Optimise convolutional Kepler model params. """
        # Unpack observations.
        self.observations = observations

        # Check kernel config.
        self._kernel_usage_check()

        # Set fixed params.
        self.fixed_params = fixed_params

        # Set priors.
        self.priors = priors

        # Set starting position as Gaussian ball around initial guess.
        # Ensure ball is within priors.
        try:
            assert all(self.initial_spread < prior[1] - prior[0]
                       for prior in self.priors.values())
            initial_position = [guess + self.initial_spread * np.random.randn(len(
                guess)) for i in range(walkers)]
        except AssertionError as err:
            raise AssertionError(
                'Initialisation spread in param space is larger than range of '
                'one or more priors. Decrease value of attribute initial_spread.')

        if not self.debug:

            # Multi-processing on all cores.
            print('Using {} cores.'.format(multiprocessing.cpu_count()))
            with multiprocessing.Pool() as pool:

                # Affine invariant MCMC ensemble sampler.
                print('Burn in.')
                sampler = emcee.EnsembleSampler(
                    walkers, len(guess), self._mcmc_lnprob_convolutional_model, pool=pool)
                pos, prob, state = sampler.run_mcmc(initial_position, burn, progress=True)
                time.sleep(2)

                print('\nWalking.')
                sampler.reset()
                sampler.run_mcmc(pos, run, progress=True)
                print('Finished MCMC.')

        else:
            sampler = emcee.EnsembleSampler(
                walkers, len(guess), self._mcmc_lnprob_convolutional_model, pool=None)
            pos, prob, state = sampler.run_mcmc(initial_position, burn)
            sampler.reset()
            sampler.run_mcmc(pos, run)

        # Get results.
        mcmcm_samples = sampler.get_chain(flat=True)
        mean_acceptance_fraction = np.mean(sampler.acceptance_fraction)
        print('Mean acceptance fraction={}.\n'.format(mean_acceptance_fraction))

        # Check convergence
        tau = sampler.get_autocorr_time(tol=0)
        autocorr = np.mean(tau)
        convergence_status = {
            'mean_tau': autocorr,
            'threshold': mcmcm_samples.shape[0] / walkers / 50}
        print('Convergence stats: mean tau={} <? threshold N/50={}'.format(
            convergence_status['mean_tau'], convergence_status['threshold']))

        # Optimized parameter results taken to be 50th percentile.
        # Errors, 16th - 84th percentiles.
        param_labels = []
        if self.kernel_type == 'TopHat':
            param_labels = ['TimeOfPeriastron', 'Eccentricity', 'RVSemiAmplitude',
                            'ArgumentOfPeriastron', 'RVOffset', 'PhotosphericRadius',
                            'IonisationRadius', 'WindTerminalVelocity']

        elif self.kernel_type == 'LogGaussian':
            param_labels = ['TimeOfPeriastron', 'Eccentricity', 'RVSemiAmplitude',
                            'ArgumentOfPeriastron', 'RVOffset', 'LogSigmaTime',
                            'MeanEmissivity', 'WindTerminalVelocity']

        elif self.kernel_type == 'RTInterp':
            if self.interp_dims == 'Point':
                param_labels = ['TimeOfPeriastron', 'Eccentricity', 'RVSemiAmplitude',
                                'ArgumentOfPeriastron']
            elif self.interp_dims == 'Line':
                param_labels = ['TimeOfPeriastron', 'Eccentricity', 'RVSemiAmplitude',
                                'ArgumentOfPeriastron', 'MassLossRate']
            elif self.interp_dims == 'Grid':
                param_labels = ['TimeOfPeriastron', 'Eccentricity', 'RVSemiAmplitude',
                                'ArgumentOfPeriastron', 'MassLossRate', 'WindTerminalVelocity']

        percentiles = calc_mcmc_chain_percentiles(mcmcm_samples, param_labels)
        print(percentiles['16th_percentile'], '\n')
        print(percentiles['Median'], '\n')
        print(percentiles['84th_percentile'])

        return mcmcm_samples, convergence_status

    def _kernel_usage_check(self):
        """ Check kernel type is compatible with observation to model. """
        if self.kernel_type == 'TopHat':
            if not len(self.observations) == 1:
                raise ValueError('Top hat kernel is not designed to co-fit '
                                 'more than one spectral line at a time.'
                                 'Reduce the number of observations.')

        elif self.kernel_type == 'LogGaussian':
            if not len(self.observations) == 1:
                raise ValueError('Log Gaussian kernel is not designed to co-fit '
                                 'more than one spectral line at a time.'
                                 'Reduce the number of observations.')

        elif self.kernel_type == 'RTInterp':
            pass

        else:
            raise NameError('Kernel type not recognised.')

    def _mcmc_lnprob_convolutional_model(self, theta):
        """ Logarithmic posterior or probability given the parameters. """
        # Prior.
        lp = self._mcmc_lnprior_convolutional_model(theta)
        if not np.isfinite(lp):
            return -np.inf

        if self.debug:
            self.fig = plt.figure(figsize=(
                len(self.observations) * 3, 4))

        # Posterior: sum of log-likelihoods.
        prob = lp
        for line, data in self.observations.items():

            # Set line data.
            self._line_label = line
            self._line_data = data

            # Likelihood.
            ll = self._mcmc_lnlike_convolutional_model(theta)
            if not np.isfinite(ll):
                return -np.inf

            # Add to posterior.
            prob += ll

        if self.debug:
            print('ln(posterior): ', prob)
            plt.tight_layout()
            plt.show()
            self._line_index = 1

        return prob

    def _mcmc_lnprior_convolutional_model(self, theta):
        """ Logarithmic prior - up to a constant. """
        if self.kernel_type == 'TopHat':

            # Top hat kernel priors.
            t0, e, k, w, c, pr, ir, v = theta
            if self.priors['TimeOfPeriastron'][0] < t0 < self.priors['TimeOfPeriastron'][1] \
                    and self.priors['Eccentricity'][0] < e < self.priors['Eccentricity'][1] \
                    and self.priors['RVSemiAmplitude'][0] < k < self.priors['RVSemiAmplitude'][1] \
                    and self.priors['ArgumentOfPeriastron'][0] < w < self.priors['ArgumentOfPeriastron'][1] \
                    and self.priors['RVOffset'][0] < c < self.priors['RVOffset'][1] \
                    and self.priors['PhotosphericRadius'][0] < pr < self.priors['PhotosphericRadius'][1] \
                    and self.priors['IonisationRadius'][0] < ir < self.priors['IonisationRadius'][1] \
                    and self.priors['WindTerminalVelocity'][0] < v < self.priors['WindTerminalVelocity'][1]:
                return 0.0

        elif self.kernel_type == 'LogGaussian':

            # Log Gaussian kernel priors.
            t0, e, k, w, c, ls, mu, v = theta
            if self.priors['TimeOfPeriastron'][0] < t0 < self.priors['TimeOfPeriastron'][1] \
                    and self.priors['Eccentricity'][0] < e < self.priors['Eccentricity'][1] \
                    and self.priors['RVSemiAmplitude'][0] < k < self.priors['RVSemiAmplitude'][1] \
                    and self.priors['ArgumentOfPeriastron'][0] < w < self.priors['ArgumentOfPeriastron'][1] \
                    and self.priors['RVOffset'][0] < c < self.priors['RVOffset'][1] \
                    and self.priors['LogSigmaTime'][0] < ls < self.priors['LogSigmaTime'][1] \
                    and self.priors['MeanEmissivity'][0] < mu < self.priors['MeanEmissivity'][1] \
                    and self.priors['WindTerminalVelocity'][0] < v < self.priors['WindTerminalVelocity'][1]:
                return 0.0

        elif self.kernel_type == 'RTInterp':

            # Radiative transfer interpolation kernel priors.
            if self.interp_dims == 'Point':
                t0, e, k, w = theta
                if self.priors['TimeOfPeriastron'][0] < t0 < self.priors['TimeOfPeriastron'][1] \
                        and self.priors['Eccentricity'][0] < e < self.priors['Eccentricity'][1] \
                        and self.priors['RVSemiAmplitude'][0] < k < self.priors['RVSemiAmplitude'][1] \
                        and self.priors['ArgumentOfPeriastron'][0] < w < self.priors['ArgumentOfPeriastron'][1]:
                    return 0.0
            elif self.interp_dims == 'Line':
                t0, e, k, w, m = theta
                if self.priors['TimeOfPeriastron'][0] < t0 < self.priors['TimeOfPeriastron'][1] \
                        and self.priors['Eccentricity'][0] < e < self.priors['Eccentricity'][1] \
                        and self.priors['RVSemiAmplitude'][0] < k < self.priors['RVSemiAmplitude'][1] \
                        and self.priors['ArgumentOfPeriastron'][0] < w < self.priors['ArgumentOfPeriastron'][1] \
                        and self.priors['MassLossRate'][0] < m < self.priors['MassLossRate'][1]:
                    return 0.0
            elif self.interp_dims == 'Grid':
                t0, e, k, w, m, v = theta
                if self.priors['TimeOfPeriastron'][0] < t0 < self.priors['TimeOfPeriastron'][1] \
                        and self.priors['Eccentricity'][0] < e < self.priors['Eccentricity'][1] \
                        and self.priors['RVSemiAmplitude'][0] < k < self.priors['RVSemiAmplitude'][1] \
                        and self.priors['ArgumentOfPeriastron'][0] < w < self.priors['ArgumentOfPeriastron'][1] \
                        and self.priors['MassLossRate'][0] < m < self.priors['MassLossRate'][1] \
                        and self.priors['WindTerminalVelocity'][0] < v < self.priors['WindTerminalVelocity'][1]:
                    return 0.0

        return -np.inf

    def _mcmc_lnlike_convolutional_model(self, theta):
        """ Logarithmic likelihood. """
        # High time resolution *equally spaced* epochs are required for easy convolution.
        # Additionally calculate two periods allowing for finite size of kernel to produce
        # at least an entire period within the np.convolve 'valid' regime i.e complete
        # signal overlap.

        if self.kernel_type == 'TopHat':

            # Current position of walkers in each dimension of parameter space.
            t0, e, k, w, c, pr, ir, v = theta

            # Configure params and epochs and kernel.
            windy_star.__init__()
            windy_star.configure_params(
                period=self.fixed_params['Period'], eccentricity=e, rv_semi_amplitude=k,
                argument_of_periastron=w, rv_offset=c,
                photospheric_radius=pr, ionisation_radius=ir,
                stellar_radius_a=self.fixed_params['StellarRadius'], wind_velocity_a=v)
            windy_star.configure_epochs(epochs=np.linspace(0, 2, self.epoch_density))
            windy_star.configure_kernel(kernel_mode=self.kernel_type)

        elif self.kernel_type == 'LogGaussian':

            # Current position of walkers in each dimension of parameter space.
            t0, e, k, w, c, ls, mu, v = theta

            # Configure params and epochs and kernel.
            windy_star.__init__()
            windy_star.configure_params(
                period=self.fixed_params['Period'], eccentricity=e, rv_semi_amplitude=k,
                argument_of_periastron=w, rv_offset=c,
                log_sigma_time=ls, mean_emissivity=mu,
                stellar_radius_a=self.fixed_params['StellarRadius'], wind_velocity_a=v)
            windy_star.configure_epochs(epochs=np.linspace(0, 2, self.epoch_density))
            windy_star.configure_kernel(kernel_mode=self.kernel_type)

        elif self.kernel_type == 'RTInterp':

            if self.interp_dims == 'Point':
                # Current position of walkers in each dimension of parameter space.
                t0, e, k, w = theta
                # t0 = self.fixed_params['TimeOfPeriastron']

                # Configure params and epochs and kernel.
                windy_star.__init__()
                windy_star.configure_params(
                    period=self.fixed_params['Period'], eccentricity=e, rv_semi_amplitude=k,
                    argument_of_periastron=w, rv_offset=.0,
                    m_dot_a=self.fixed_params['MassLossRate'],
                    wind_velocity_a=self.fixed_params['WindTerminalVelocity'])
                windy_star.configure_epochs(epochs=np.linspace(0, 2, self.epoch_density))
                windy_star.configure_kernel(kernel_mode=self.kernel_type,
                                            kernel_line=self._line_label,
                                            interp_dims=self.interp_dims,
                                            emissivity_interpolators=self.emissivity_obj)
            elif self.interp_dims == 'Line':
                # Current position of walkers in each dimension of parameter space.
                t0, e, k, w, m = theta

                # Configure params and epochs and kernel.
                windy_star.__init__()
                windy_star.configure_params(
                    period=self.fixed_params['Period'], eccentricity=e, rv_semi_amplitude=k,
                    argument_of_periastron=w, rv_offset=.0,
                    m_dot_a=m, wind_velocity_a=self.fixed_params['WindTerminalVelocity'])
                windy_star.configure_epochs(epochs=np.linspace(0, 2, self.epoch_density))
                windy_star.configure_kernel(kernel_mode=self.kernel_type,
                                            kernel_line=self._line_label,
                                            interp_dims=self.interp_dims,
                                            emissivity_interpolators=self.emissivity_obj)
            elif self.interp_dims == 'Grid':
                # Current position of walkers in each dimension of parameter space.
                t0, e, k, w, m, v = theta

                # Configure params and epochs and kernel.
                windy_star.__init__()
                windy_star.configure_params(
                    period=self.fixed_params['Period'], eccentricity=e,
                    rv_semi_amplitude=k, argument_of_periastron=w, rv_offset=.0,
                    m_dot_a=m, wind_velocity_a=v)
                windy_star.configure_epochs(epochs=np.linspace(0, 2, self.epoch_density))
                windy_star.configure_kernel(kernel_mode=self.kernel_type,
                                            kernel_line=self._line_label,
                                            interp_dims=self.interp_dims,
                                            emissivity_interpolators=self.emissivity_obj)

        # Calculate centroid velocity of convolved/time averaged Keplerian orbit.
        res_df = windy_star.convolutional_model_centroid_velocity

        # Use second period only, for reasons commented above.
        res_df = res_df.loc[res_df['Phase'] > 1.0]
        res_df['Phase'] -= 1.0

        # Interpolate epochs for unevenly sampled observations.
        self._line_data['Phase'] = ((self._line_data['JD'] - t0)
                                    % self.fixed_params['Period']) \
                                   / self.fixed_params['Period']
        cv_interp = np.interp(self._line_data['Phase'].values,
                              res_df['Phase'],
                              res_df['ConvolvedVelocity'])

        # Log likelihood: Chi-squared + ln(2 * pi * sigma^2).
        ln_like = -0.5 * ((((cv_interp - self._line_data['Velocity']) ** 2)
                           / (self._line_data['VelocityError'] ** 2))
                          + np.log(2 * np.pi * (self._line_data['VelocityError'] ** 2))).sum()

        if self.debug:
            ax = self.fig.add_subplot(1, len(self.observations), self._line_index)
            ax.scatter(self._line_data['Phase'], self._line_data['Velocity'], c='red', s=2)
            ax.scatter(res_df['Phase'], res_df['ConvolvedVelocity'], c='orange', s=2)
            ax.scatter(self._line_data['Phase'], cv_interp, c='blue', s=2)
            ax.set_title('ln(likelihood)={}'.format(round(ln_like, 5)), fontsize=10)
            ax.set_xlabel('Orbital Phase $\phi$', fontsize=10)
            if self._line_index == 1:
                ax.set_ylabel('Velocity / $\\rm{km\,s^{-1}}$', fontsize=10)
            self._line_index += 1

        return ln_like
