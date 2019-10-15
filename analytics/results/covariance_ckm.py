import corner
import matplotlib.pyplot as plt

from ckm.core.utils import unpickle_mcmc_results


# Setup
MCMC_PICKLE = 'ConvolutionalRTInterpStaticModel_example_target_datetime.p'

# Unpack MCMC analytics.
fixed_params, sample_chain, systemic, convergence, observations = unpickle_mcmc_results(MCMC_PICKLE)

# Histograms and covariance.
corner.corner(sample_chain, bins=50, quantiles=[0.16, 0.5, 0.84],
              labels=['$T_{0}$', '$e$', 'k', '$\omega$'],
              show_titles=True, title_kwargs={"fontsize": 10})
plt.show()
