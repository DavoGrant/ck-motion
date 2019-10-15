import numpy as np

from ckm.config import OBSERVATIONS_PATH
from ckm.optimise.mcmc_convolutional import MCMCOptimiser
from ckm.radiative.emissivity import Emissivity
from ckm.core.utils import unpickle_observations, pickle_mcmc_results


# Setup
DEBUG = False
OBSERVATIONS = [{'line': 1, 'rv_offset': -21.0,
                 'data': ['emission_line_fits_1.p']},
                {'line': 2, 'rv_offset': -25.0,
                 'data': ['emission_line_fits_2.p']}]


# Optimise lines individually.
for o in OBSERVATIONS:

    # Load observations.
    obs = {}
    label = 'EtaCar_CrossValidation_{}'.format(o['line'])
    raw_obs = unpickle_observations(
        OBSERVATIONS_PATH, o['data'])

    # Pre-set velocity offsets as per Kepler model.
    raw_obs['Velocity'] -= o['rv_offset']
    obs[o['line']] = raw_obs

    # Load interpolation objects.
    emissivity = Emissivity(setup_mode=False, pickle_dir='interp_jar/rt_code_v1')

    # Optimise.
    op = MCMCOptimiser(
        kernel_type='RTInterp', interp_dims='Point',
        emissivity_obj=emissivity, epoch_density=1000)
    op.debug = DEBUG
    op.initial_spread = 1e-6
    simulation_chain, convergence = op.optimise_convolutional_model(
        observations=obs,
        fixed_params={'Period': 1000,
                      'WindTerminalVelocity': 1300,
                      'MassLossRate': 1e-5},
        guess=np.array([2555555, 0.60, 60, 250]),
        priors={'TimeOfPeriastron': (2555535, 2555575),
                'Eccentricity': (0.10, 0.90),
                'RVSemiAmplitude': (10, 120),
                'ArgumentOfPeriastron': (180, 360)},
        walkers=128, burn=6000, run=6000)

    # Pickle results.
    pickle_dict = {
        'fixed': op.fixed_params,
        'chain': simulation_chain,
        'convergence': convergence,
        'observations': op.observations}
    pickle_mcmc_results(pickle_dict, 'ConvolutionalRTInterpStatic', label)
