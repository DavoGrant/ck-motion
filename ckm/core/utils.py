import os
import pickle
import datetime
import numpy as np
import pandas as pd


def convert_km_per_s_to_au_per_day(kms):
    """ Convert velocities from km/sec to au/day. """
    return kms / 1.496e+8 * (3600 * 24)


def convert_time_to_phase_units(days, period):
    """ Convert durations of time from days to phase. """
    return days / period * (2 * np.pi)


def unpickle_observations(obs_path, obs_list, time_of_periastron=None,
                          period=None, n_periods=1):
    """ Load observations from pickle. """
    obs = pd.DataFrame()
    for p in obs_list:
        obs_i = pickle.load(open(os.path.join(obs_path, p), 'rb'))

        if time_of_periastron is not None and period is not None:
            obs_i['Phase'] = ((obs_i['JD'] - time_of_periastron) % period) / period
            obs_i = obs_i.sort_values(
                by=['Phase'], ascending=True).reset_index(drop=True)
            extension = pd.DataFrame()
            for r in range(n_periods - 1):
                repeat = obs_i.copy()
                repeat['Phase'] += r + 1
                extension = pd.concat([extension, repeat], axis=0, ignore_index=True)
                obs_i = pd.concat([obs_i, extension], axis=0, ignore_index=True)
        obs = pd.concat([obs, obs_i], axis=0, ignore_index=True)

    return obs


def pickle_mcmc_results(results, model, observations):
    """ Write pickled dictionary of MCMC results. """
    directory = 'mcmc_jar'
    unique_time = str(datetime.datetime.utcnow()).replace(
        '-', '').replace(' ', '_').replace(':', '')[:15]
    pickle_path = os.path.join(
        directory, '{}Model_{}_{}.p'.format(
            model, observations, unique_time))
    if not os.path.exists(directory):
        os.mkdir(directory)
    pickle.dump(results, open(pickle_path, 'wb'))


def unpickle_mcmc_results(mcmc_pickle, reference_periastron=None,
                          period=None, n_periods=1, directory='mcmc_jar'):
    """ Load MCMC results from pickle. """
    pickle_dict = pickle.load(open(os.path.join(
        directory, mcmc_pickle), 'rb'))
    fixed_params = pickle_dict['fixed']
    sample_chain = pickle_dict['chain']
    convergence = pickle_dict['convergence']
    observations = pickle_dict['observations']

    if reference_periastron is not None and period is not None:
        if isinstance(observations, dict):
            for line, line_obs in observations.items():
                line_obs = calc_phase_and_duplicate_periods(
                    line_obs, reference_periastron, period, n_periods)
                observations[line] = line_obs
        else:
            observations = calc_phase_and_duplicate_periods(
                observations, reference_periastron, period, n_periods)

    try:
        systemic = pickle_dict['systemic']
        return fixed_params, sample_chain, systemic, convergence, observations
    except KeyError as err:
        # No systemic velocity available.
        return fixed_params, sample_chain, convergence, observations


def calc_phase_and_duplicate_periods(obs, ref_peri, p, np):
    """ Calculate phase adn duplicate periods. """
    obs['Phase'] = ((obs['JD'] - ref_peri) % p) / p
    obs = obs.sort_values(by=['Phase'], ascending=True).reset_index(drop=True)
    extension = pd.DataFrame()
    for r in range(np - 1):
        repeat = obs.copy()
        repeat['Phase'] += r + 1
        extension = pd.concat([extension, repeat], axis=0, ignore_index=True)
    return pd.concat([obs, extension], axis=0, ignore_index=True)


def calc_mcmc_chain_percentiles(chain, param_labels):
    """ Calculate the 16th, 50th, 84th percentiles. """
    # Optimized parameter results taken to be 50th percentile.
    # Errors, 16th - 84th percentiles.
    optimized_parameters_median = np.percentile(chain, 50, axis=0)
    optimized_parameters_16pc = np.percentile(chain, 16, axis=0)
    optimized_parameters_84pc = np.percentile(chain, 84, axis=0)
    mcmc_percentiles = pd.DataFrame(
        np.column_stack((
            optimized_parameters_median,
            optimized_parameters_16pc,
            optimized_parameters_84pc)),
        index=param_labels,
        columns=['Median', '16th_percentile', '84th_percentile'])

    return mcmc_percentiles
