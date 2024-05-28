import numpy as np
# Algorithms
algo_ver = 'exp_runtime'
run_in_p = False
n_cores = 6
solver_verbose = True

methods = ['Algorithm.SPOForest',
           'Algorithm.SPOForest',
           'Algorithm.SPOForest',
           ]
methods_classes = ['SPOForest',
                   'SPOForest',
                   'SPOForest', ]

methods_names = ['SPO-RF', 'PRF-EMP', 'PRF-NORM']

ids = [0,1,2]

methods_dict = {'packages': [methods[m] for m in ids],
                'name': [methods_names[m] for m in ids],
                'classes': [methods_classes[m] for m in ids]
                }
# Parameters for each algorithm
algo_params_dict = {}
tree_depth = np.Inf
n_trees = 200
min_t_weights = 5
discret_quant = 0.01
algo_verbose = False
p_model_no_saa = True
method_iptbrf = 'mode'
iptbrf_predict_verbose = True
sub_sample_ratio = 0.8

forest_predict_options_dict = {'n_way': 2,
                               0: {'no_saa': True,
                                          'method': 'mean',
                                          'verbose': False},
                               1: {'no_saa': True,
                                        'method': 'mode',
                                        'verbose': True},
                               }

algo_params_dict['PRF-EMP'] = {'model_params':
                                       {'pred_mode': 'IPTB-RF',
                                        'max_depth': tree_depth,
                                        'n_estimators': n_trees,
                                        'min_tweights_per_node': min_t_weights,
                                        'quant_discret': discret_quant,
                                        'max_features': 'auto',
                                        'SPO_weight_param': 1,
                                        'SPO_full_error': True,
                                        'PDO': True,
                                        'sampling_ratio': sub_sample_ratio,
                                        },
                                   'fit_params': {'verbose': algo_verbose},
                                   'predict_params': forest_predict_options_dict,
                                   'return_loc': 'decision',
                                   'fit_pres_data': True
                                   }

algo_params_dict['PRF-NORM'] = {'model_params':
                                        {'pred_mode': 'IPTB-RF',
                                         'max_depth': tree_depth,
                                         'n_estimators': n_trees,
                                         'min_tweights_per_node': min_t_weights,
                                         'quant_discret': discret_quant,
                                         'max_features': 'auto',
                                         'SPO_weight_param': 1,
                                         'SPO_full_error': False,
                                         'PDO': True,
                                         'sampling_ratio': sub_sample_ratio,
                                         },
                                    'fit_params': {'verbose': algo_verbose},
                                    'predict_params': forest_predict_options_dict,
                                    'return_loc': 'decision',
                                    'fit_pres_data': True
                                    }

algo_params_dict['SPO-RF'] = {'model_params':
                                  {'pred_mode': 'SPO-M',
                                   'max_depth': tree_depth,
                                   'n_estimators': n_trees,
                                   'min_tweights_per_node': min_t_weights,
                                   'quant_discret': discret_quant,
                                   'max_features': 'auto',
                                   'SPO_weight_param': 1,
                                   'SPO_full_error': True,
                                   'PDO': True,
                                   },
                              'fit_params': {'verbose': algo_verbose},
                              'predict_params': {'no_saa': True,
                                                 'n_way': 1,
                                                 },
                              'return_loc': 'cost',
                              'fit_pres_data': True
                              }


