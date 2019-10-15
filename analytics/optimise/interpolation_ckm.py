import numpy as np

from config import RT_DB_PATH
from ckm.radiative.emissivity import Emissivity


# Fixed config.
INTERP_LABEL = 'rt_code_v1'
WIND_VELOCITY_SOL = 'beta_1'
INITIAL_VELOCITY = 100
STELLAR_RADIUS = 10
SONIC_RADIUS = 11
RT_TABLE = 'rt_table'

# Grid parameterisations: must be evenly spaced and correspond
# to available param space output from radiative transfer. N.B
# limiting times provides a kernel size which ensures complete
# overlap in the convolution's valid mode for a given
# number/fraction of signal/periods.
Mdot = 1e-5
Vinf = 1300
Time = np.linspace(0, 10, 100)

# Iterate spectral lines.
spectral_lines = [1, 2]
for line in spectral_lines:

    # Instantiate emissivity object in setup mode.
    # N.B see line_label for verbose transition name.
    emissivity_setup = Emissivity(setup_mode=True)
    emissivity_setup.rt_results_db = RT_DB_PATH
    emissivity_setup.rt_results_table = RT_TABLE

    # Setup spectral line and wind velocity profile.
    emissivity_setup.set_spectral_line(line)
    emissivity_setup.set_wind_velocity_profile(
        wind_velocity_solution=WIND_VELOCITY_SOL,
        initial_velocity=INITIAL_VELOCITY,
        stellar_radius=STELLAR_RADIUS,
        sonic_radius=SONIC_RADIUS)

    # Build interpolation object.
    emissivity_setup.build_static_interpolation_object(Mdot, Vinf, Time)

    # Write interpolation object to disk.
    emissivity_setup.pickle_interpolation_object(INTERP_LABEL)
