import math
import numpy as np
import matplotlib.pyplot as plt

from ckm.core.motion import windy_star
from ckm.radiative.emissivity import Emissivity
from ckm.core.utils import calc_mcmc_chain_percentiles, \
    unpickle_mcmc_results


# Setup
MCMC_PICKLE = 'ConvolutionalRTInterpStaticModel_example_target_datetime.p'
INTERP_PICKLE = 'interp_jar/rt_code_v1'
N_CHAIN_SAMPLES = 100
N_PERIODS = 2
PERIOD = 1000
REFERENCE_PERIASTRON = 2555555
EPOCH_DENSITY = 500

# Unpack MCMC analytics.
fixed_params, sample_chain, systemic, convergence, observations = unpickle_mcmc_results(
    MCMC_PICKLE, REFERENCE_PERIASTRON, PERIOD, N_PERIODS)

# Convergence stats.
print('Convergence stats: mean tau={} <? threshold N/50={}'.format(
    convergence['mean_tau'], convergence['threshold']))

# Optimized parameter results taken to be 50th percentile.
# Errors, 16th - 84th percentiles.
param_labels = ['TimeOfPeriastron', 'Eccentricity',
                'RVSemiAmplitude', 'ArgumentOfPeriastron']
percentiles = calc_mcmc_chain_percentiles(sample_chain, param_labels)
print(percentiles['16th_percentile'], '\n')
print(percentiles['Median'], '\n')
print(percentiles['84th_percentile'])

# Load interpolation objects.
emissivity = Emissivity(setup_mode=False, pickle_dir=INTERP_PICKLE)

# Iterate observed spectral lines.
fig = plt.figure(figsize=(20, 10))
line_index = 1
for line, data in observations.items():
    print('Line {}.'.format(line_index))

    ax = fig.add_subplot(2, math.ceil(len(observations) / 2), line_index)
    markers, caps, bars = ax.errorbar(
        data['Phase'], data['Velocity'] + systemic[line_index - 1], yerr=None,
        color='#000000', fmt='o', markersize='1', elinewidth=1,
        capsize=1, capthick=1, label='Observations')
    [bar.set_alpha(0.2) for bar in bars]
    [cap.set_alpha(0.2) for cap in caps]

    # Plot model samples.
    count = 0
    for t0, e, k, w in sample_chain[np.random.randint(
            len(sample_chain), size=N_CHAIN_SAMPLES), :]:
        try:
            windy_star.__init__()
            windy_star.configure_params(
                period=fixed_params['Period'], eccentricity=e, rv_semi_amplitude=k,
                argument_of_periastron=w, rv_offset=systemic[line_index - 1],
                m_dot_a=fixed_params['MassLossRate'],
                wind_velocity_a=fixed_params['WindTerminalVelocity'])
            windy_star.configure_epochs(epochs=np.linspace(0, N_PERIODS, EPOCH_DENSITY))
            windy_star.configure_kernel(kernel_mode='RTInterp',
                                        kernel_line=line, interp_dims='Point',
                                        emissivity_interpolators=emissivity)
            res_df = windy_star.convolutional_model_centroid_velocity
            adjusted_phase = res_df['Phase'] + ((t0 - REFERENCE_PERIASTRON) / PERIOD)
            if count == 0:
                ax.plot([], [],
                        color='#4891dc', alpha=1, linewidth=1,
                        label='MCMC CK (RTInterp) solution samples')
                ax.legend(loc='upper right', fontsize=8)
            else:
                ax.plot(adjusted_phase, res_df['ConvolvedVelocity'],
                        color='#4891dc', alpha=0.2, linewidth=0.3,
                        label=None)
        except ValueError as e:
            continue
        finally:
            count += 1
            print('{}/{}'.format(count, N_CHAIN_SAMPLES))

    ax.set_xlabel('Orbital Phase $\phi$', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    if line_index == 1 or line_index == math.ceil(len(observations) / 2) + 1:
        ax.set_ylabel('Velocity / $\\rm{km\,s^{-1}}$', fontsize=10)
    line_index += 1
    ax.set_xlim(0.84, 1.16)

plt.tight_layout()
plt.show()
