import corner
import matplotlib.pyplot as plt

from ckm.core.utils import unpickle_mcmc_results


# Setup
MCMC_PICKLE = 'KeplerModel_example_target_line_datetime[1,2,3,4]_20190701_133356.p'

# Unpack MCMC analytics.
fixed_params, sample_chain, convergence, observations = unpickle_mcmc_results(MCMC_PICKLE)

# Histograms and covariance.
corner.corner(sample_chain, bins=50, quantiles=[0.16, 0.5, 0.84],
              labels=['$T_{0}$', '$e$', 'k', '$\omega$', '$\gamma$'],
              show_titles=True, title_kwargs={"fontsize": 10})
plt.show()
