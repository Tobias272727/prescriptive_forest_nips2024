problem_loc = 'Decision_Problem_Solver.single_machine'
obj_func = 'CompletionTime'
from .poly_data_config import *

data_params['release'] = True
data_params['sorted'] = False
data_params['normalise'] = False
data_params['obj_func'] = obj_func

config_ver = '1_cj_release_' + data_type


n_job = 10
n_ins = n_job

from .ml_config import *