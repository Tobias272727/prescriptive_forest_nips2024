"""
SPO RANDOM FOREST IMPLEMENTATION

This code will work for general predict-then-optimize applications. Fits SPO Forest to dataset of feature-cost pairs.

The structure of the decision-making problem of interest should be encoded in a file called decision_problem_solver.py. 
Specifically, this code requires two functions:
  get_num_decisions(): returns number of decision variables (i.e., dimension of cost vector for underlying decision problem)
  find_opt_decision(): returns for a matrix of cost vectors the corresponding optimal decisions for those cost vectors 
"""
import copy
from scipy.sparse import csr_matrix
import numpy as np
from .mtp import MTP
from .SPO_tree_greedy import SPOTree
from collections import Counter
from scipy.spatial import distance
from joblib import Parallel, delayed
from Algorithm.tree_utils import max_elements

class SPOForest(object):
    """
    This function initializes the SPO forest

    FOREST PARAMETERS:

    n_estimators: number of SPO trees in the random forest

    max_features: number of features to consider when looking for the best split in each node

    run_in_parallel, num_workers: if run_in_parallel is set to True, enables parallel computing among num_workers threads.
    If num_workers is not specified, uses the number of cpu cores available. The task of computing each SPO tree in the forest
    is distributed among the available cores. (each tree may only use 1 core and thus this arg is set to None in SPOTree class)

    TREE PARAMETERS (DIRECTLY PASSED TO SPOTree CLASS):

    max_depth: maximum training depth of each tree in the forest (default = Inf: no depth limit)

    min_weight_per_node: the mininum number of observations (with respect to cumulative weight) per node for each tree in the forest

    quant_discret: continuous variable split points are chosen from quantiles of the variable corresponding to quant_discret,2*quant_discret,3*quant_discret, etc..

    SPO_weight_param: splits are decided through loss = SPO_loss*SPO_weight_param + MSE_loss*(1-SPO_weight_param)
        SPO_weight_param = 1.0 -> SPO loss
        SPO_weight_param = 0.0 -> MSE loss (i.e., CART)

    SPO_full_error: if SPO error is used, are the full errors computed for split evaluation,
        i.e. are the alg's decision losses subtracted by the optimal decision losses?

    Keep all other parameter values as default
    """
    def __init__(self, n_estimators=10,
                 pred_mode='SPO-M',
                 sampling_ratio=1, **kwargs):
        self.n_estimators = n_estimators
        # if (run_in_parallel == False):
        #     num_workers = 1
        # if num_workers is None:
        #      #this uses all available cpu cores

        # the kwargs for DT training
        self.decision_kwargs = None
        self.sampling_ratio = sampling_ratio

        # Parameters related to the decision-making problem
        if pred_mode == 'IPTB-RF':
            # this parameter will be passed to the trees
            kwargs['pred_mode'] = 'IPTB'
            self.sampling = 'bootstrap'
            self.PDO_flag = True
            self.solver = kwargs['solver']
        elif pred_mode == 'IPTB-boosting':
            # Use scenario reduction based boosting(SRB) to do sampling and then train IPTB model to prescribe decisions
            kwargs['pred_mode'] = 'IPTB'
            self.sampling = 'SRB'
            self.PDO_flag = True
            self.solver = kwargs['solver']
        elif pred_mode == 'SPO-M':
            # use SPO-F to do the predict then optimise
            self.PDO_flag = False
            self.sampling = 'bootstrap'
            kwargs['pred_mode'] = 'SPO-M'
            self.solver = kwargs['solver']

        self.fitted_Y = None
        self.fitted_mu = None

        # ESTABLISHING SPOTREES
        self.forest = [None] * n_estimators
        for t in range(n_estimators):
            self.forest[t] = SPOTree(**kwargs)

    """
    This function fits the SPO forest on data (X, C, weights).
    
    X: The feature data used in tree splits. Can either be a pandas data frame or numpy array, with:
      (a) rows of X = observations
      (b) columns of X = features
    C: the cost vectors used in the leaf node models. Must be a numpy array, with:
      (a) rows of C = observations
      (b) columns of C = cost vector components
    weights: a numpy array of case weights. Is 1-dimensional, with weights[i] yielding weight of observation i
    feats_continuous: If False, all feature are treated as categorical. If True, all feature are treated as continuous.
      feats_continuous can also be a boolean vector of dimension = num_features specifying how to treat each feature
    verbose: if verbose=True, prints out progress in tree fitting procedure
    verbose_forest: if verbose_forest=True, prints out progress in the forest fitting procedure
    seed: seed for rng
    """
    def fit(self, X, C,
            sigma_mat=None,
            # weights=None, verbose_forest=False, seed=None,
            # feats_continuous=False, verbose=False, refit_leaves=False,
            **kwargs):

        # PARAMS INITIALIZATION
        self.decision_kwargs = kwargs
        self.fitted_Y = C
        if 'mu' in kwargs.keys():
            self.fitted_mu = kwargs['mu']
        if 'D' in kwargs.keys():
            self.fitted_d = kwargs['D']
        # todo update a input seed for random forest
        np.random.seed(1)

        if 'weights' in kwargs.keys():
            weights = kwargs['weights']
        else:
            weights = np.ones([X.shape[0]])


        tree_seeds = np.random.randint(0, high=2**18-1, size=self.n_estimators)

        # FIT THE MODEL
        # note: the parallel running is by tree model itself, which is transited by kwargs
        for t in range(self.n_estimators):
            self.forest[t] = _fit_tree(t,
                                       self.n_estimators,
                                       self.forest[t],
                                       X, C,
                                       weights,
                                       tree_seeds[t],
                                       sampling=self.sampling,
                                       sampling_ratio=self.sampling_ratio,
                                       **kwargs)

    """
    Prints all trees in the forest
    Required: call forest fit() method first
    """
    def traverse(self):
        for t in range(self.n_estimators):
            print("Printing Tree " + str(t+1) + "out of " + str(self.n_estimators))
            self.forest[t].traverse()
            print("\n\n\n")

    """
    Predicts decisions or costs given data Xnew
    Required: call tree fit() method first
    
    method: method for aggregating decisions from each of the individual trees in the forest. Two approaches:
      (1) "mean": averages predicted cost vectors from each tree, then finds decision with respect to average cost vector
      (2) "mode": each tree in the forest estimates an optimal decision; take the most-recommended decision
    
    NOTE: return_loc argument not supported:
    (If return_loc=True, est_decision will also return the leaf node locations for the data, in addition to the decision.)
    """
    def est_decision(self, Xnew, no_saa=False, verbose=False, method="mean"):
        if self.PDO_flag:
            forest_decisions = np.zeros([Xnew.shape[0], self.solver.get_num_decisions()])
            num_obs = Xnew.shape[0]
            if method == "mean":
                # record each tree decisions
                for i in range(num_obs):
                    # merge all the samples into sample_mat
                    sample_mat = np.array([])
                    for t in range(self.n_estimators):
                        subs, loc_tuple = self.forest[t].tree.predict(Xnew[i, :].reshape([1,-1]),
                                                                      np.array(range(0, Xnew.shape[0])),
                                                                      return_loc=True)
                        tree_id = loc_tuple[0][0]
                        sub = subs[0]
                        # get data slice and stack them to sample mat
                        data_slice = self.forest[t].tree.tree[tree_id].fitted_model.Y
                        sample_mat = np.vstack([sample_mat, data_slice]) if sample_mat.size else data_slice
                    # since the samples from trees are bootstrapped, trace the original data from fitted_Y and fitted_mu of
                    # the forest model also get the weight for each sample:
                    # if sample_mat[t, :] identical to another sample_mat[t,:], then the weight_mat[t] = 2
                    # else weight_mat[t] = 1

                    weight_mat = np.ones([sample_mat.shape[0]])
                    index_list = []
                    for t in range(sample_mat.shape[0]):
                        # Look for the same sample row in fitted_Y (use first one, no difference)
                        index = np.where(np.all(self.fitted_Y == sample_mat[t, :], axis=1))[0]
                        # in mu mat, the id will be the same
                        index_list.append(index[0])
                        mu_vals = self.fitted_mu[index[0], :]
                        # for j in range(t+1, sample_mat.shape[0]):
                        #     if np.array_equal(sample_mat[t, :], sample_mat[j, :]):
                        #         weight_mat[t] += 1
                        #         weight_mat[j] = 0
                    #
                    # # remove the weight in weight_mat == 0 and sample in sample_mat
                    # index_list.remove(index_list[weight_mat != 0])
                    # sample_mat = sample_mat[weight_mat != 0, :]
                    # weight_mat = weight_mat[weight_mat != 0]

                    # if no_saa, use the best decision according to mu_mat:
                    if no_saa:
                        new_mu = self.fitted_mu[index_list, :]
                        new_mu = new_mu[:, index_list]
                        new_d = self.fitted_d[index_list, :]
                        d_idx = max_elements(new_mu, weights=weight_mat)
                        forest_decisions[i, :] = new_d[d_idx, :]
                        # otherwise, use the SAA to get the decision
                    else:
                        forest_decisions[i, :] = self.solver.solve_multiple_models(sample_mat,
                                                                                   weight_mat=weight_mat,
                                                                                   PDO_flag=True)['weights']
            else:
                tree_decisions = np.zeros([Xnew.shape[0], self.n_estimators, self.solver.get_num_decisions()])
                for t in range(self.n_estimators):
                    prediction, loc_tuple \
                        = self.forest[t].tree.predict(Xnew, np.array(range(0, Xnew.shape[0])), return_loc=True)
                    # Since IPTB, we use the SAA to estimate the decision prediction is the mat of tree.sub
                    d_list = []
                    for i, sub, tree_id in zip(range(num_obs), prediction, loc_tuple[0]):
                        d = self.forest[t].tree.tree[tree_id].fitted_model.decision
                        tree_decisions[i, t, :] = d
                # get each row's mode decision
                for i in range(num_obs):
                    # use the voting method to get decision of each row
                    count_var = Counter(map(tuple, tree_decisions[i, :, :])).most_common()
                    if count_var[0][1] == 1:
                        if verbose:
                            print("all tree gives different decision, plz check the model")
                        # all tree has one vote, use mean to get the decision
                        forest_decisions[i, :] = self.est_decision(Xnew[i, :].reshape([1, -1]),
                                                                   method='mean',
                                                                   no_saa=no_saa)
                    else:
                        forest_decisions[i, :] = np.array(count_var[0][0])
        else:
            if method == "mean":
                forest_costs = self.est_cost(Xnew)
                forest_decisions = self.OR_dict.solve_multiple_models(forest_costs,
                                                                      **self.decision_kwargs)['weights']

            elif method == "mode":
                num_obs = Xnew.shape[0]
                tree_decisions = [None]*self.n_estimators
                for t in range(self.n_estimators):
                    tree_decisions[t] = self.forest[t].est_decision(Xnew)
                tree_decisions = np.array(tree_decisions)
                forest_decisions = np.zeros((num_obs,tree_decisions.shape[2]))
                for i in range(num_obs):
                    forest_decisions[i] = _get_mode_row(tree_decisions[:, i, :])

        return forest_decisions

    def predict(self, Xnew):
        tree_costs = [None]*self.n_estimators
        for t in range(self.n_estimators):
            tree_costs[t] = self.forest[t].est_cost(Xnew)
        tree_costs = np.array(tree_costs)
        forest_costs = np.mean(tree_costs, axis=0)
        return forest_costs


"""
Helper methods (ignore)
"""
def _fit_tree(t, n_estimators, tree,
              X, C,
              weights, tree_seed,
              sampling='random',
              sampling_ratio=1,
              sigma_mat=None,
              verbose_forest=False, **kwargs):
    """
    Do the sampling then fit the tree model
    """
    if verbose_forest:
        print("Fitting tree " + str(t+1) + "out of " + str(n_estimators))

    # bootstraping of the data
    num_obs = C.shape[0]

    np.random.seed(tree_seed)
    if sampling == 'bootstrap':
        bootstrap_inds = np.random.choice(range(num_obs), size=int(num_obs * sampling_ratio), replace=True)
    else:
        # Scenario reduction
        if sigma_mat is None:
            for i in range(num_obs):
                sigma_mat[i, :] = np.linalg.norm(C - C[i, :], axis=1)

        # find the lowest sigma value
        sigma_min = np.min(sigma_mat)
        min_pos = np.where(sigma_mat == np.min(sigma_mat))[0]

    Xb = np.copy(X[bootstrap_inds])
    Cb = np.copy(C[bootstrap_inds])

    # Check the data in kwargs that need to be devided
    kwargs_temp = kwargs.copy()
    devide_list = ['A', 'D', 'mu']
    for d in devide_list:
        if d in kwargs.keys():
            # if 1-d array, then just reshape it by the bootstrap_inds
            if len(kwargs[d].shape) == 1 or (kwargs[d].shape[0] == num_obs and kwargs[d].shape[1] == 1):
                kwargs_temp[d] = kwargs[d][bootstrap_inds]
            # if 2-d array and both dim are identical to the observation nums, then sort both by bootstrap_inds
            elif len(kwargs[d].shape) == 2 and kwargs[d].shape[0] == num_obs and kwargs[d].shape[1] == num_obs:
                temp_data = copy.deepcopy(kwargs[d])
                # get the rows by bootstrap_inds
                temp_data = temp_data[bootstrap_inds, :]
                temp_data = temp_data[:, bootstrap_inds]
                # get the cols by bootstrap_inds
                kwargs_temp[d] = temp_data
            elif len(kwargs[d].shape) == 2 and kwargs[d].shape[0] == num_obs and kwargs[d].shape[1] != num_obs:
                kwargs_temp[d] = kwargs[d][bootstrap_inds, :]

    kwargs_temp['seed'] = tree_seed
    kwargs_temp['weights'] = np.copy(weights[bootstrap_inds])

    tree.fit(Xb, Cb, **kwargs_temp)
    return tree


def _get_mode_row(a):
    count_var = Counter(map(tuple, a)).most_common()
    return np.array(count_var[0][0])