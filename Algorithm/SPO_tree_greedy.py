"""
SPO GREEDY TREE IMPLEMENTATION

This code will work for general predict-then-optimize applications.
Fits SPO (greedy) tree to dataset of feature-cost pairs.

"""
import time
from scipy.sparse import csr_matrix
import numpy as np
from .mtp import MTP
from scipy.spatial import distance


class SPOTree:
    """
    This function initializes the SPO tree

    Parameters:
    #MUST HAVE#
    solver: instance of any solver class that identical to DPS.BaseSolver class


    #OPTIONAL#
    pred_mode:
    max_depth: maximum training depth of each tree in the forest (default = Inf: no depth limit)
    min_weight_per_node: the mininum number of observations (with respect to cumulative weight) per node for each tree in the forest
    quant_discret: continuous variable split points are chosen from quantiles of the variable corresponding to quant_discret,2*quant_discret,3*quant_discret, etc..

    SPO_weight_param: splits are decided through loss = SPO_loss*SPO_weight_param + MSE_loss*(1-SPO_weight_param)
      SPO_weight_param = 1.0 -> SPO loss
      SPO_weight_param = 0.0 -> MSE loss (i.e., CART)

    SPO_full_error: if SPO error is used, are the full errors computed for split evaluation,
      i.e. are the alg's decision losses subtracted by the optimal decision losses?

    run_in_parallel: if set to True, enables parallel computing among num_workers threads. If num_workers is not
    specified, uses the number of cpu cores available.

    max_features: number of features to consider when looking for the best split in each node. Useful when building random forests. Default equal to total num features

    Keep all other parameter values as default
    """

    def __init__(self, **kwargs):
        """
        Initialisation of the SPO tree mode
        """
        # params
        self.SPO_weight_param = kwargs["SPO_weight_param"]
        self.SPO_full_error = kwargs["SPO_full_error"]

        self.solver = kwargs["solver"]
        self.tree = MTP(**kwargs)
        # pre-define the fit_kwargs, which will be updated in self.fit() function

        self.fit_kwargs = None
        self.pruned = False

    def fit(self, X, C,
            **kwargs):
        """
            This function fits the tree on data (X,C,weights).

            X: The feature data used in tree splits. Can either be a pandas data frame or numpy array, with:
              (a) rows of X = observations
              (b) columns of X = features

            C: the cost vectors used in the leaf node models. Must be a numpy array, with:
              (a) rows of C = observations
              (b) columns of C = cost vector components

            A: the pre-calculated objective functions

            D: the pre-calculated decision corresponding to input C's

            weights: a numpy array of case weights. Is 1-dimensional, with weights[i] yielding weight of observation i
            feats_continuous: If False, all feature are treated as categorical. If True, all feature are treated as continuous.
              feats_continuous can also be a boolean vector of dimension = num_features specifying how to treat each feature

            verbose: if verbose=True, prints out progress in tree fitting procedure

            Keep all other parameter values as default
        """

        # Initialize params
        self.pruned = False  # if the tree is pruned
        self.fit_kwargs = kwargs
        if 'A' in kwargs.keys() and kwargs['A'] is None:
            if self.SPO_full_error and self.SPO_weight_param != 0.0:
                # If SPO weight != 0, decision should be made by solver.
                kwargs['A'] = self.solver.solve_multiple_models(C, **kwargs)['objective'][0]

        # USING 1.0 LOSS BOUND
        # TODO update these codes.
        SPO_loss_bound = 1.0
        MSE_loss_bound = 1.0
        # if self.SPO_weight_param != 0.0 and self.SPO_weight_param != 1.0:
        #     if self.SPO_full_error == True:
        #         SPO_loss_bound = -float("inf")
        #         for i in range(num_ins):
        #             SPO_loss = -self.OR_dict.solve_multiple_models(self._get_or_c(C,i).reshape(1, -1),
        #             **kwargs)['objective'][0] - A[i]
        #             if SPO_loss >= SPO_loss_bound:
        #                 SPO_loss_bound = SPO_loss
        #
        #     else:
        #         c_max = np.max(C, axis=0)
        #         SPO_loss_bound = -self.OR_dict.solve_multiple_models(-c_max.reshape(1, -1), **kwargs)['objective'][0]
        #
        #     # Upper bound for MSE loss: maximum pairwise difference between any two elements
        #     dists = distance.cdist(C, C, 'sqeuclidean')
        #     MSE_loss_bound = np.max(dists)
        # else:
        #     SPO_loss_bound = 1.0
        #     MSE_loss_bound = 1.0

        # if mode is IPTB, check input mu and D
        if self.tree.pred_mode == 'IPTB':
            if 'mu' not in kwargs.keys() or 'D' not in kwargs.keys():
                raise ValueError("mu and D must be provided for IPTB mode")
            mu = kwargs['mu']
            D = kwargs['D']
            if mu is None or D is None:
                raise ValueError("mu and D must be provided for IPTB mode")
            #
            # mu = mu.astype(float)
            # D = D.astype(float)
            # mu = csr_matrix(mu)
            # D = csr_matrix(D)
            # kwargs['mu'] = mu
            # kwargs['D'] = D

        self.tree.fit(X, C,
                      SPO_loss_bound=SPO_loss_bound,
                      MSE_loss_bound=MSE_loss_bound,
                      **kwargs)

    def traverse(self, verbose=False):
        """
        Prints out the tree.
        Required: call tree fit() method first
        Prints pruned tree if prune() method has been called, else prints unpruned tree
        verbose=True prints additional statistics within each leaf
        """
        self.tree.traverse(verbose=verbose)

    def prune(self, Xval, Cval,
              weights_val=None, one_SE_rule=True, verbose=False, approx_pruning=False):
        """
        Prunes the tree. Set verbose=True to track progress
        """
        num_obs = Cval.shape[0]

        Aval = np.array(range(num_obs))
        if self.SPO_full_error is True and self.SPO_weight_param != 0.0:
            for i in range(num_obs):
                Aval[i] = self.solver.solve_multiple_models(Cval[i, :].reshape(1, -1),
                                                             **self.fit_kwargs)['objective'][0]

        self.tree.prune(Xval, Aval, Cval,
                        weights_val=weights_val, one_SE_rule=one_SE_rule, verbose=verbose,
                        approx_pruning=approx_pruning)
        self.pruned = True

    """
    Produces decision or cost given data Xnew
    Required: call tree fit() method first
    Uses pruned tree if pruning method has been called, else uses unpruned tree
    Argument alpha controls level of pruning. If not specified, uses alpha trained from the prune() method
    
    As a step in finding the estimated decisions for data (Xnew), this function first finds
    the leaf node locations corresponding to each row of Xnew. It does so by a top-down search
    starting at the root node 0. 
    If return_loc=True, est_decision will also return the leaf node locations for the data, in addition to the decision.
    """

    def est_decision(self, Xnew,
                     alpha=None,
                     return_loc=False,
                     no_saa=False,
                     verbose=True):
        # estimate the decision of tree mmodel
        if self.tree.pred_mode == 'IPTB':
            prediction, loc_tuple \
                = self.tree.predict(Xnew, np.array(range(0, Xnew.shape[0])),
                                    alpha=alpha, return_loc=True)

            # Since IPTB, we use the SAA to estimate the decision prediction is the mat of tree.sub
            d_list = []
            for sub, tree_id in zip(prediction, loc_tuple[0]):
                if self.tree.tree[tree_id].fitted_model.saa_optimised or no_saa:
                    saa_d = self.tree.tree[tree_id].fitted_model.decision
                else:
                    data_slice = self.tree.tree[tree_id].fitted_model.Y
                    saa_d = self.solver.solve_multiple_models(data_slice, PDO_flag=True)['weights']
                    self.tree.tree[tree_id].fitted_model.saa_optimised = True
                    self.tree.tree[tree_id].fitted_model.decision = saa_d
                d_list.append(saa_d.reshape([-1]))

            # Make it a d-array
            prediction = np.array(d_list)
            return prediction
        else:
            return self.tree.predict(Xnew, np.array(range(0, Xnew.shape[0])),
                                     alpha=alpha,
                                     return_loc=return_loc)

    def predict(self, Xnew, alpha=None, return_loc=False):
        return self.tree.predict(Xnew, np.array(range(0, Xnew.shape[0])),
                                 alpha=alpha, return_loc=return_loc,
                                 get_cost=True)

    """
    Other methods (ignore)
    """

    def get_tree_encoding(self, x_train=None):
        return self.tree.get_tree_encoding(x_train=x_train)

    def get_pruning_alpha(self):
        if self.pruned:
            return self.tree.alpha_best
        else:
            return 0
