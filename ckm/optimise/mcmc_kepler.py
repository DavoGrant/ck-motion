import time
import emcee
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from ckm.core.motion import windy_star
from ckm.core.utils import calc_mcmc_chain_percentiles


class MCMCOptimiser(object):
    """ Markov Chain Monte Carlo class."""

    def __init__(self):
        self.debug = False
        self.observations = None
        self.fixed_params = None
        self.initial_spread = 1e-3
        self.priors = None

    def __repr__(self):
        return 'Markov Chain Monte Carlo Optimiser Object for ' \
               'Keplerian centroid velocity model.'

    def optimise_kepler_model(self, observations, fixed_params, guess, priors,
                              walkers=64, burn=500, run=500):
        """ Optimise Kepler model params. """
        # Unpack observations.
        self.observations = observations

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
                    walkers, len(guess), self._mcmc_lnprob_kepler_model, pool=pool)
                pos, prob, state = sampler.run_mcmc(initial_position, burn, progress=True)
                time.sleep(2)

                print('\nWalking.')
                sampler.reset()
                sampler.run_mcmc(pos, run, progress=True)
                print('Finished MCMC.')

        else:
            sampler = emcee.EnsembleSampler(
                walkers, len(guess), self._mcmc_lnprob_kepler_model, pool=None)
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
        param_labels = ['TimeOfPeriastron', 'Eccentricity', 'RVSemiAmplitude',
                        'ArgumentOfPeriastron', 'RVOffset']
        percentiles = calc_mcmc_chain_percentiles(mcmcm_samples, param_labels)
        print(percentiles['16th_percentile'], '\n')
        print(percentiles['Median'], '\n')
        print(percentiles['84th_percentile'])

        return mcmcm_samples, convergence_status

    def _mcmc_lnprob_kepler_model(self, theta):
        """ Logarithmic posterior or probability given the parameters. """
        # Prior.
        lp = self._mcmc_lnprior_kepler_model(theta)
        if not np.isfinite(lp):
            return -np.inf

        # Likelihood.
        ll = self._mcmc_lnlike_kepler_model(theta)
        if not np.isfinite(ll):
            return -np.inf

        # Posterior.
        prob = lp + ll

        return prob

    def _mcmc_lnprior_kepler_model(self, theta):
        """ Logarithmic prior - up to a constant. """
        # Current position of walkers in each dimension of parameter space.
        t0, e, k, w, c = theta

        # Check prior.
        if self.priors['TimeOfPeriastron'][0] < t0 < self.priors['TimeOfPeriastron'][1] \
                and self.priors['Eccentricity'][0] < e < self.priors['Eccentricity'][1] \
                and self.priors['RVSemiAmplitude'][0] < k < self.priors['RVSemiAmplitude'][1] \
                and self.priors['ArgumentOfPeriastron'][0] < w < self.priors['ArgumentOfPeriastron'][1] \
                and self.priors['RVOffset'][0] < c < self.priors['RVOffset'][1]:
            return 0.0

        return -np.inf

    def _mcmc_lnlike_kepler_model(self, theta):
        """ Logarithmic likelihood. """
        # Current position of walkers in each dimension of parameter space.
        t0, e, k, w, c = theta

        # Configure params and epochs.
        windy_star.__init__()
        windy_star.configure_params(
            period=self.fixed_params['Period'], eccentricity=e,
            rv_semi_amplitude=k, argument_of_periastron=w, rv_offset=c)
        windy_star.configure_epochs(
            epochs=self.observations['JD'].values, time_of_periastron=t0)

        # Calculate centroid velocity of a binary Keplerian orbit.
        res_df = windy_star.keplerian_model_centroid_velocity

        # Log likelihood: Chi-squared + ln(2 * pi * sigma^2).
        ln_like = -0.5 * ((((res_df['KeplerVelocity'] - self.observations['Velocity']) ** 2)
                          / (self.observations['VelocityError'] ** 2))
                          + np.log(2 * np.pi * (self.observations['VelocityError'] ** 2))).sum()

        if self.debug:
            plt.scatter(windy_star.epochs, self.observations['Velocity'], c='red', s=2)
            plt.scatter(res_df['Phase'], res_df['KeplerVelocity'], c='blue', s=2)
            plt.title('ln(likelihood)={}'.format(round(ln_like, 5)), fontsize=10)
            plt.xlabel('Orbital Phase $\phi$', fontsize=10)
            plt.ylabel('Velocity / $\\rm{km\,s^{-1}}$', fontsize=10)
            plt.tight_layout()
            plt.show()

        return ln_like
