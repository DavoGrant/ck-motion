import numpy as np
import matplotlib.pyplot as plt

from config import OBSERVATIONS_PATH
from ckm.core.motion import windy_star
from ckm.core.utils import unpickle_observations


# Setup
OBSERVATIONS = ['emission_line_fits_1.p', 'emission_line_fits_2.p']

# Load observations.
obs = unpickle_observations(
    OBSERVATIONS_PATH, OBSERVATIONS,
    time_of_periastron=2555555, period=1000, n_periods=2)

# Configure params and epochs.
windy_star.configure_params(
    period=1000, eccentricity=0.5, rv_semi_amplitude=50,
    argument_of_periastron=250, rv_offset=-20)
windy_star.configure_epochs(epochs=np.linspace(0, 2, 500))

# Calculate centroid velocity of a binary Keplerian orbit.
res_df = windy_star.keplerian_model_centroid_velocity

# Plot.
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(obs['Phase'], obs['Velocity'], s=2, c='#000000')
ax1.plot(res_df['Phase'], res_df['KeplerVelocity'], linewidth=2)
ax1.set_xlabel('Orbital Phase $\phi$', fontsize=14)
ax1.set_ylabel('Velocity / $\\rm{km\,s^{-1}}$', fontsize=14)
plt.tight_layout()
plt.show()
