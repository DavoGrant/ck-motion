import numpy as np
import matplotlib.pyplot as plt

from ckm.core.motion import windy_star
from ckm.core.utils import calc_mcmc_chain_percentiles, \
    unpickle_mcmc_results


# Setup
MCMC_PICKLE = 'KeplerModel_example_target_line_datetime.p'
N_CHAIN_SAMPLES = 100
N_PERIODS = 2
PERIOD = 1000
REFERENCE_PERIASTRON = 2555555

# Unpack MCMC analytics.
fixed_params, sample_chain, convergence, observations = unpickle_mcmc_results(
    MCMC_PICKLE, REFERENCE_PERIASTRON, PERIOD, N_PERIODS)

# Convergence stats.
print('Convergence stats: mean tau={} <? threshold N/50={}'.format(
    convergence['mean_tau'], convergence['threshold']))

# Optimized parameter results taken to be 50th percentile.
# Errors, 16th - 84th percentiles.
param_labels = ['TimeOfPeriastron', 'Eccentricity', 'RVSemiAmplitude',
                'ArgumentOfPeriastron', 'RVOffset']
percentiles = calc_mcmc_chain_percentiles(sample_chain, param_labels)
print(percentiles['16th_percentile'], '\n')
print(percentiles['Median'], '\n')
print(percentiles['84th_percentile'])

# Plot observed data.
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 1, 1)
markers, caps, bars = ax1.errorbar(
    observations['Phase'], observations['Velocity'], yerr=None,
    color='#000000', fmt='o', markersize='1', elinewidth=1,
    capsize=1, capthick=1, label='Observations')
[bar.set_alpha(0.2) for bar in bars]
[cap.set_alpha(0.2) for cap in caps]

# Plot model samples.
count = 0
for t0, e, k, w, c in sample_chain[np.random.randint(
        len(sample_chain), size=N_CHAIN_SAMPLES), :]:
    try:
        windy_star.__init__()
        windy_star.configure_params(
            period=fixed_params['Period'], eccentricity=e, rv_semi_amplitude=k,
            argument_of_periastron=w, rv_offset=c)
        windy_star.configure_epochs(epochs=np.linspace(0, N_PERIODS, 2000 * N_PERIODS))
        res_df = windy_star.keplerian_model_centroid_velocity
        adjusted_phase = res_df['Phase'] + ((t0 - REFERENCE_PERIASTRON) / PERIOD)
        if count == 0:
            ax1.plot([], [],
                     color='#4891dc', alpha=1, linewidth=1,
                     label='MCMC Kepler solution samples')
            ax1.legend(loc='upper right', fontsize=10)
        else:
            ax1.plot(adjusted_phase, res_df['KeplerVelocity'],
                     color='#4891dc', alpha=0.2, linewidth=0.3,
                     label=None)
    except ValueError as e:
        continue
    finally:
        count += 1
        print('{}/{}'.format(count, N_CHAIN_SAMPLES))

ax1.set_xlabel('Orbital Phase $\phi$', fontsize=10)
ax1.set_ylabel('Velocity / $\\rm{km\,s^{-1}}$', fontsize=10)
ax1.tick_params(axis='both', labelsize=10)

plt.tight_layout()
plt.show()
