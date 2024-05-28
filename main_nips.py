# data formats
import numpy as np
from sklearn.model_selection import train_test_split

# system
import os
import importlib
# utils
from utils.exp_runner import exp_runner
from utils.data_reader import read_sms_data, sampling_from_sms_data

# config files: modify this to switch between different configurations
from nips_configs.synthetic.tj import *

# use importlib to import solve and gen_data from problem_loc
module = importlib.import_module(problem_loc)
Solver = getattr(module, 'Solver')
gen_data = getattr(module, 'gen_data')

if __name__ == "__main__":
    # get the training and testing samples
    if 'pcb' in data_type:
        param_dict = gen_data(n_job, **data_params)
        # data of pcb
        data_path = 'data/sms_data/PCB_jobs.xlsx'
        x, y, sorting_array, col_idx_list = read_sms_data(data_path)

        # split x,y into testing_samples and training_samples
        # return the index and get sorting_array_train, sorting_array_test
        x_train, x_test, y_train, y_test, sorting_array_train, sorting_array_test = train_test_split(x, y,
                                                                                                     sorting_array,
                                                                                                     test_size=testing_ratio,
                                                                                                     random_state=0)
        # Get the testing samples
        # use x_test, y_test to get the testing samples
        test_po_x, test_po_y, test_p_x, test_p_y, test_sorting_id_data, feat_continous \
            = sampling_from_sms_data(x_test, y_test, sorting_array_test, col_idx_list,
                                     n_ins=test_size,
                                     seed=test_seed,
                                     instance_size=n_ins,
                                     )
    elif 'synthetic' in data_type:
        # data of random
        param_dict, test_p_x, test_p_y, test_po_x, test_po_y = gen_data(n_job,
                                                                        n_sample=test_size,
                                                                        seed=test_seed,
                                                                        **data_params)
        feat_continous = None

    # loop each n_size setting to get the final sols
    for e, size in enumerate(exp_n_sizes):
        if 'pcb' in data_type:
            # get the train data
            train_po_x, train_po_y, train_p_x, train_p_y, train_sorting_id_data, feat_continous \
                = sampling_from_sms_data(x_train, y_train, sorting_array_train, col_idx_list,
                                         n_ins=size,
                                         seed=train_seed,
                                         instance_size=n_job,
                                         )
        elif 'synthetic' in data_type:
            # data of random
            param_dict123, train_p_x, train_p_y, train_po_x, train_po_y = gen_data(n_job,
                                                                                   n_sample=size,
                                                                                   seed=train_seed,
                                                                                   **data_params)

        # data path and outcome path
        # make a folder if config folder does not exist
        if not os.path.exists("output/nips_npy/" + config_ver):
            os.makedirs("output/nips_npy/" + config_ver)
        if not os.path.exists("output/nips_npy/" + config_ver + "/algo_" + algo_ver):
            os.makedirs("output/nips_npy/" + config_ver + "/algo_" + algo_ver)

        data_str = "output/nips_npy/" + config_ver + "/" + "n_size_" + str(size)
        outcome_str = "output/nips_npy/" + config_ver + "/algo_" + algo_ver + "n_size_" + str(size)
        data_str_test = "output/nips_npy/" + config_ver + "/" + "n_size_" + str(test_size)

        """
        Get the historical data
        """
        # Check if instances is calculated or need recompute
        try:
            testing_PM_data_dict = np.load(data_str_test + "_testing.npy",
                                           allow_pickle=True).item()
            test_p_y = testing_PM_data_dict['Y_test']
            test_p_x = testing_PM_data_dict['X_test']
            param_dict = testing_PM_data_dict['param_dict']
            print("his testing data loaded from database")
        except FileNotFoundError:

            print("his testing data did not found, start calculating")
            # If the file doesn't exist, generate the data using the function
            pm_solver = Solver(param_dict, n_core=n_cores)
            dict_result_testing = pm_solver.solve_multiple_models(test_p_y, verbose=False)

            testing_PM_data_dict = {}
            testing_PM_data_dict['param_dict'] = param_dict
            testing_PM_data_dict['obj'] = dict_result_testing['objective']
            testing_PM_data_dict['decision'] = dict_result_testing['weights']
            for y_row, opt_d, opt_obj in zip(test_p_y, testing_PM_data_dict['decision'], testing_PM_data_dict['obj']):
                if pm_solver.get_obj(y_row, opt_d) != opt_obj:
                    print(pm_solver.get_obj(y_row, opt_d), opt_obj)
            testing_PM_data_dict['Y_test'] = test_p_y
            testing_PM_data_dict['X_test'] = test_p_x
            np.save(data_str_test + "_testing.npy", testing_PM_data_dict)

        try:
            training_PM_data_dict = np.load(data_str + "_training.npy",
                                            allow_pickle=True).item()
            print("his training data loaded from database")
        except FileNotFoundError:
            print("his training data did not found, start calculating")
            # If the file doesn't exist, generate the data using the function
            pm_solver = Solver(param_dict, n_core=n_cores)
            training_PM_data_dict = {}
            dict_result_training = pm_solver.solve_multiple_models(train_p_y)
            training_PM_data_dict['obj'] = dict_result_training['objective']
            training_PM_data_dict['decision'] = dict_result_training['weights']

            # mu_mat for pre-calculated objective functions
            mu_mat = np.zeros([train_p_y.shape[0], train_p_y.shape[0]])
            for i in range(train_p_y.shape[0]):
                d = training_PM_data_dict['decision'][i]
                mu_mat[i, :] = [pm_solver.get_obj(y, d) for y in train_p_y]
            training_PM_data_dict['mu_mat'] = mu_mat
            np.save(data_str + "_training.npy", training_PM_data_dict)

        # Load from training and testing scheduling data
        obj_mat = training_PM_data_dict["obj"]
        output_list = [obj_mat]
        d_array = training_PM_data_dict["decision"]
        mu_mat = training_PM_data_dict["mu_mat"]
        testing_decision = testing_PM_data_dict["decision"]

        # Model training
        print('#######################################################################################')
        print('\n Exp_started for config: ', config_ver, 'n_size: ', size, 'algo_ver: ', algo_ver)

        # a test code:
        # get the best solution from historical decision for test instances.
        regret = 0
        pm_solver = Solver(param_dict,
                           n_core=n_cores,
                           time_limit=30)

        for y_row, opt_d in zip(test_p_y, testing_decision):
            obj_vals = [pm_solver.get_obj(y_row, d) for d in d_array]
            # calculate the regret of the best solution
            regret += (np.min(obj_vals) - pm_solver.get_obj(y_row, opt_d)) / pm_solver.get_obj(y_row, opt_d)
        print('the regret of the best IPTB solution is: ', regret / train_p_y.shape[0])

        # Run the experiments
        # for po: train the model by train_po_x, train_po_y, but test the model by test_p_x, test_p_y
        input_po_data_dict = {'opt_d_test': testing_decision,
                              'test_X': test_p_x,
                              'test_Y': test_p_y,
                              'train_X': train_po_x,
                              'train_Y': train_po_y}

        # for p: train and test the model by train_p_x, train_p_y
        input_p_data_dict = {'opt_d_test': testing_decision,
                             'test_X': test_p_x,
                             'test_Y': test_p_y,
                             'train_X': train_p_x,
                             'train_Y': train_p_y}

        for algo in methods_dict['name']:
            if algo_params_dict[algo]['fit_pres_data']:
                algo_params_dict[algo]['model_params']['solver'] = pm_solver
                algo_params_dict[algo]['fit_params']['mu'] = mu_mat
                algo_params_dict[algo]['fit_params']['feats_continuous'] = feat_continous
                algo_params_dict[algo]['fit_params']['D'] = d_array
                algo_params_dict[algo]['fit_params']['A'] = obj_mat

        exp_runner(methods_dict,
                   algo_params_dict,
                   input_po_data_dict,
                   input_p_data_dict,
                   pm_solver,
                   results_path=outcome_str,
                   verbose=True,
                   save_model=False,
                   save_path=None)
