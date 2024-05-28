import numpy as np
import importlib
import time




def exp_runner(algo_dict: dict,
               params_dict: dict,
               input_po_data: dict, input_p_data: dict,
               solver: object,
               results_path=None,
               verbose=False,
               save_model=False,
               save_path=None, ):
    """
    This function runs the experiments for the given algorithms and parameters on the given data.
    algo_list: list of algorithms to run, the name has been specified in the config file and link to the algorithm
    ___ inputs:
    params_dict: dictionary of parameters for each algorithm, the key is the algorithm name, value is another dictionary
    as input for the algorithm
    data: the data to run the experiments on
    ___ outputs:
    print the results of the experiments
    try to save the trained models by pickle
    save the results in a npy file
    """
    algo_packages = algo_dict['packages']
    algo_classes = algo_dict['classes']
    algo_list = algo_dict['name']
    # load the str in algo_list to the actual class
    num_ins = input_p_data['test_Y'].shape[0]

    # count the number of methods
    n_output = 0
    for algo in algo_list:
        n_output +=params_dict[algo]['predict_params']['n_way']


    spo_loss_mat = np.zeros([num_ins, n_output])
    obj_loss_mat = np.zeros([num_ins, n_output])
    pred_obj_mat = np.zeros([num_ins, n_output])
    run_times_mat = np.zeros([num_ins, len(algo_list)])

    # use importlib to import the class
    algo_count = 0
    output_count = 0
    for algo, class_name, pac in zip(algo_list, algo_classes, algo_packages):
        if verbose: print('experimenting with ', algo, ' data size: ', input_p_data['train_Y'].shape[0])
        algo_class = importlib.import_module(pac)
        algo_class = getattr(algo_class, class_name)

        # run the algorithm with the given parameters
        algo_instance = algo_class(**params_dict[algo]['model_params'])
        start_time = time.time()
        # fit p or po model with different data
        if params_dict[algo]['fit_pres_data']:
            algo_instance.fit(input_p_data['train_X'], input_p_data['train_Y'],
                              **params_dict[algo]['fit_params'])
        else:
            algo_instance.fit(input_po_data['train_X'], input_po_data['train_Y'],
                              **params_dict[algo]['fit_params'])
        fit_end_time = time.time()
        run_times_mat[:, algo_count] = fit_end_time - start_time
        # save the model if specified
        if save_model:
            try:
                algo_instance.save_model(save_path)
            except:
                print('Error: model not saved')

        # 3. predict and save results
        for o in range(params_dict[algo]['predict_params']['n_way']):
            # predict the decision for each predict param dict
            count = 0
            MAPE = 0
            for y_row, x_row, opt_d in zip(input_p_data['test_Y'], input_p_data['test_X'], input_p_data['opt_d_test']):
                if params_dict[algo]['return_loc'] == 'decision':
                    d = algo_instance.est_decision(x_row.reshape([1, -1]),
                                                   **params_dict[algo]['predict_params'][o])
                else:
                    # algo don't have est_decision method, have a prediction method
                    # get the po_test_row first, this is a 2D array with shape (n_test_size, n_features)
                    if input_p_data['train_X'].shape == input_po_data['train_X'].shape:
                        po_test_row = x_row.reshape(1, -1)
                    else:
                        po_test_row = x_row.reshape([input_p_data['test_Y'].shape[1], -1])
                    # todo if X has a fix feature num, use po data instead,
                    pred_val = algo_instance.predict(po_test_row)
                    # GET MAPE
                    MAPE += np.mean(np.abs(pred_val - y_row) / y_row)
                    d = solver.solve_multiple_models(pred_val.reshape([1, -1]))['weights']

                pred_obj = solver.get_obj(y_row, d.reshape(-1))
                opt_obj = solver.get_obj(y_row, opt_d)
                spo_loss_mat[count, output_count] = (pred_obj - opt_obj) / opt_obj
                obj_loss_mat[count, output_count] = (pred_obj - opt_obj)
                pred_obj_mat[count, output_count] = pred_obj
                count += 1

            # if verbose print the results
            if verbose:
                print('results of ', algo, ' with output ', o)
                print('prediction MAPE: ', MAPE / num_ins)
                print('Regret percentage: ', spo_loss_mat[:, output_count].sum() / num_ins)
                print('Objective loss: ', obj_loss_mat[:, output_count].sum() / num_ins)
                print('Objective prediction: ', pred_obj_mat[:, output_count].sum() / num_ins)
                print('Run time: ', run_times_mat[:, algo_count].sum() / num_ins)
            # next output
            output_count += 1
        algo_count += 1

    result_dict = {'spo_loss_mat': spo_loss_mat,
                   'obj_loss_mat': obj_loss_mat,
                   'pred_obj_mat': pred_obj_mat,
                   'run_times_mat': run_times_mat,}

    # 4. save the results
    np.save(results_path + '_results.npy', result_dict)
    return result_dict
