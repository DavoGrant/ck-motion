import numpy as np
import matplotlib.pyplot as plt

from config import OBSERVATIONS_PATH
from ckm.core.motion import windy_star
from ckm.radiative.emissivity import Emissivity
from ckm.core.utils import unpickle_observations


# Setup
OBSERVATIONS = ['emission_line_fits_1.p', 'emission_line_fits_2.p']

# Load observations.
obs = unpickle_observations(
    OBSERVATIONS_PATH, OBSERVATIONS,
    time_of_periastron=2555555, period=1000, n_periods=2)

# Load interpolation objects.
emissivity = Emissivity(setup_mode=False, pickle_dir='interp_jar/rt_code_v1')

# Configure params and epochs and kernel.
windy_star.configure_params(
    period=1000, eccentricity=0.7, rv_semi_amplitude=60,
    argument_of_periastron=250, rv_offset=-20)
windy_star.configure_epochs(epochs=np.linspace(0, 2, 1000))
windy_star.configure_kernel(kernel_mode='RTInterp',
                            kernel_line=4, interp_dims='Point',
                            emissivity_interpolators=emissivity)

# Calculate centroid velocity of convolved/time averaged Keplerian orbit
res_df = windy_star.convolutional_model_centroid_velocity

# Plot.
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(obs['Phase'], obs['Velocity'], s=2, c='#000000')
ax1.plot(res_df['Phase'], res_df['ConvolvedVelocity'], linewidth=2)
ax1.set_xlabel('Orbital Phase $\phi$', fontsize=14)
ax1.set_ylabel('Velocity / $\\rm{km\,s^{-1}}$', fontsize=14)
plt.tight_layout()
plt.show()
