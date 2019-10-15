import platform


OBSERVATIONS_PATH = None
RT_PATH = None
print('Platform={}'.format(platform.node()))
if platform.node() == 'your_computer':
    OBSERVATIONS_PATH = '/observations/target_name/data_release'
    RT_DB_PATH = '/simulations/radiative_transfer/model_run/output.db'

elif platform.node() == 'different_computer':
    OBSERVATIONS_PATH = '/diff_observations/target_name/data_release'
    RT_DB_PATH = '/diff_simulations/radiative_transfer/model_run/output.db'
