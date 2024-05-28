problem_loc = 'Decision_Problem_Solver.single_machine'
obj_func = 'Tardiness'
from .poly_data_config import *

config_ver = '1_tj_release' + data_type

data_params['release'] = True
data_params['sorted'] = False
data_params['normalise'] = False
data_params['obj_func'] = obj_func

n_job = 10
n_ins = n_job

from .ml_config import *