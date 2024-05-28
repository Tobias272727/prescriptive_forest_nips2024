problem_loc = 'Decision_Problem_Solver.single_machine'
obj_func = 'CompletionTime'
from .pcb_data_config import *

config_ver = '1_cj_' + data_type

data_params['release'] = False
data_params['sorted'] = False
data_params['normalise'] = False
data_params['obj_func'] = obj_func


n_job = 20
n_ins = n_job

from .ml_config import *