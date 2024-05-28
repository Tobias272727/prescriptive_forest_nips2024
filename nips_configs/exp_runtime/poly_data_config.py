deg = 1
data_type = 'synthetic_deg_' + str(deg)

data_params = {'skip_data': False,
               'data_dis': 'poly',
               'deg': deg,}


exp_n_sizes = [200, 500, 1000, 2000]

test_size = 100
testing_ratio = 0.2
test_seed = 114514
train_seed = 1919810