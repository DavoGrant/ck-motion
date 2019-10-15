import numpy as np

from config import OBSERVATIONS_PATH
from ckm.optimise.mcmc_kepler import MCMCOptimiser
from ckm.core.utils import unpickle_observations, pickle_mcmc_results


todo_list = [('emission_line_1', 'emission_line_fits_1.p'),
             ('emission_line_2', 'emission_line_fits_2.p')]

for task in todo_list:

    # Setup
    DEBUG = False
    OBSERVATIONS_LABEL = task[0]
    OBSERVATIONS = [task[1]]

    # Load observations.
    obs = unpickle_observations(OBSERVATIONS_PATH, OBSERVATIONS)

    # Optimise.
    op = MCMCOptimiser()
    op.debug = DEBUG
    simulation_chain, convergence = op.optimise_kepler_model(
        observations=obs,
        fixed_params={'Period': 1000},
        guess=np.array([2555555, 0.60, 60, 250, 0]),
        priors={'TimeOfPeriastron': (2555535, 2555575),
                'Eccentricity': (0.10, 0.90),
                'RVSemiAmplitude': (10, 120),
                'ArgumentOfPeriastron': (180, 360),
                'RVOffset': (-100, 100)},
        walkers=128, burn=6000, run=6000)

    # Pickle results.
    pickle_dict = {
        'fixed': op.fixed_params,
        'chain': simulation_chain,
        'convergence': convergence,
        'observations': op.observations}
    pickle_mcmc_results(pickle_dict, 'Kepler', OBSERVATIONS_LABEL)
