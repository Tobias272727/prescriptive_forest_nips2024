import numpy as np

algo_ver = 'mode_rf_small_leaf'
run_in_p = False
n_cores = 6
solver_verbose = True
# Algorithms to run
methods = ['sklearn.tree',
           'sklearn.ensemble',
           'Algorithm.SAA',
           'Algorithm.SPO_tree_greedy',
           'Algorithm.SPO_tree_greedy',
           'Algorithm.SPOForest',
           'Algorithm.SPOForest',
           'Algorithm.SPOForest',
           'Algorithm.SPOForest', ]

methods_classes = ['DecisionTreeRegressor',
                   'RandomForestRegressor',
                   'SaaMethod',
                   'SPOTree',
                   'SPOTree',
                   'SPOForest',
                   'SPOForest',
                   'SPOForest',
                   'SPOForest',]

methods_names = ['CART', 'RF',
                 'SAA',
                 'IPTB-T-NORM', 'IPTB-T-EMP',
                 'IPTB-RF-NORM', 'IPTB-RF-EMP', ]

# selected_ids: control the methods to run
selected_ids = [0, 1, 2, 3, 4, 5, 6]
selected_ids = [1, 2, 5, 6]

methods_dict = {'packages': [methods[m] for m in selected_ids],
                'name': [methods_names[m] for m in selected_ids],
                'classes': [methods_classes[m] for m in selected_ids]
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

tree_predict_options_dict = {'n_way': 2,
                             0: {'no_saa': True,
                                        'verbose': False},
                             1: {'no_saa': False,
                                     'verbose': False}
                             }

forest_predict_options_dict = {'n_way': 2,
                               0: {'no_saa': True,
                                          'method': 'mean',
                                          'verbose': False},
                               # 1: {'no_saa': False,
                               #         'method': 'mean',
                               #         'verbose': False},
                               1: {'no_saa': True,
                                        'method': 'mode',
                                        'verbose': True},
                               }

algo_params_dict['IPTB-T-NORM'] = {'model_params':
                                       {'pred_mode': 'IPTB',
                                        'max_depth': tree_depth,
                                        'min_tweights_per_node': min_t_weights,
                                        'quant_discret': discret_quant,
                                        'max_features': 'all',
                                        'SPO_weight_param': 1,
                                        'SPO_full_error': True,
                                        'PDO': False,
                                        },
                                   'fit_params': {'verbose': algo_verbose},
                                   'predict_params': tree_predict_options_dict,
                                   'return_loc': 'decision',
                                   'fit_pres_data': True
                                   }

algo_params_dict['IPTB-T-EMP'] = {'model_params':
                                      {'pred_mode': 'IPTB',
                                       'max_depth': tree_depth,
                                       'min_tweights_per_node': min_t_weights,
                                       'quant_discret': discret_quant,
                                       'max_features': 'all',
                                       'run_in_parallel': run_in_p,
                                       'num_workers': n_cores,
                                       'SPO_weight_param': 1,
                                       'SPO_full_error': False,
                                       'PDO': True,
                                       },
                                  'fit_params': {'verbose': algo_verbose},
                                  'predict_params': tree_predict_options_dict,
                                  'return_loc': 'decision',
                                  'fit_pres_data': True}

algo_params_dict['IPTB-RF-EMP'] = {'model_params':
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

algo_params_dict['IPTB-RF-NORM'] = {'model_params':
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

algo_params_dict['CART'] = {'model_params':
                                {'max_depth': 100,
                                 },
                            'fit_params': {},
                            'predict_params': {},
                            'return_loc': 'cost',
                            'fit_pres_data': False
                            }

algo_params_dict['RF'] = {'model_params':
                              {'max_depth': 100,
                               'n_estimators': n_trees,
                               },
                          'fit_params': {},
                          'predict_params': {'n_way': 1,},
                          'return_loc': 'cost',
                          'fit_pres_data': False
                          }

algo_params_dict['SAA'] = {'model_params': {},
                           'fit_params': {},
                           'predict_params': {'n_way': 1,
                                              0: {'time_limit': 10},
                                              },
                           'return_loc': 'decision',
                           'fit_pres_data': True
                           }
