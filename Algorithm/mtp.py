"""
MTP: helper class for SPO_tree_greedy and SPOForest
"""
import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed
from .leaf_model import *
from .tree_utils import _is_split_legal, _is_node_large, which_child_multi, which, fast_avg, get_unique_valsNinds

"""

Model Trees for Personalization (MTP)

This is a class for training MTPs and using them for prediciton. It builds leaf modes
in each leaf corresponding to the given leaf modeling class.

MTP has the following methods:
  __init__(): initializes the MTP
  fit(): trains the MTP on data: contexts X, decisions P (labeled as A in this code), responses Y
  traverse(): prints out the learned MTP
  prune(): prunes the tree on a held-out validation set to prevent overfitting
  predict(): predict response distribution given new contexts X and decisions P

"""


class MTP(object):
    """
    This function initializes the tree

    max_depth: the maximum depth of the pre-pruned tree (default = Inf: no depth limit)

    min_weight_per_node: the mininum number of observations (with respect to cumulative weight) per node

    min_depth: the minimum depth of the pre-pruned tree (default: set equal to max_depth)

    min_diff: if depth > min_depth, stop splitting if improvement in fit does not exceed min_diff

    quant_discret: continuous variable split points are chosen from quantiles of the variable corresponding to
                    - quant_discret: 2*quant_discret, 3*quant_discret, etc..

    run_in_parallel: if set to True, enables parallel computing among num_workers threads. If num_workers is not
    specified, uses the number of cpu cores available.

    Any additional arguments are passed to the leaf_model init() function

    NOTE: the following parameters below are experimental and are not fully supported. Set equal to default values.

    binary_splits: if True, use binary splits when building the tree, else consider multiway splits
    (i.e., when splitting on a variable, split on all unique vals)

    debias_splits/frac_debias_set/min_debias_set_size: Additional params when binary_splits = True. If debias_splits = True, then in each node,
    hold out frac_debias_set of the training set (w.r.t. case weights) to evaluate the error of the best splitting point for each feature.
    Stop bias-correcting when we have insufficient data; i.e. the total weight in the debias set < min_debias_set_size.
      Note: after finding best split point, we then refit the model on all training data and recalculate the training error

    only_singleton_splits: If only_singleton_splits = False, allows categorical splits to be on subsets of values rather than singletons

    max_features: number of features to consider when looking for the best split in each node.
    Useful when building random forests. Default equal to total num features
    """

    def __init__(self, max_depth=float("inf"),
                 min_weights_per_node=15, min_depth=None, min_diff=0,
                 binary_splits=True,
                 debias_splits=False, frac_debias_set=0.2, min_debias_set_size=100,
                 quant_discret=0.01,
                 run_in_parallel=False, num_workers=None,
                 only_singleton_splits=True,
                 max_features="all",
                 *leafargs, **leafkwargs):
        # Compile tree parameters, store in "self"
        tree_params = _TreeParams()
        tree_params.max_depth = max_depth
        tree_params.min_weights_per_node = min_weights_per_node
        if min_depth is None:
            tree_params.min_depth = max_depth
        else:
            tree_params.min_depth = min_depth
        tree_params.min_diff = min_diff
        tree_params.binary_splits = binary_splits
        if (binary_splits == True):
            tree_params.debias_splits = debias_splits
        else:
            tree_params.debias_splits = False
        tree_params.frac_debias_set = frac_debias_set
        tree_params.min_debias_set_size = min_debias_set_size
        tree_params.quant_discret = quant_discret

        # run_in_parallel=True
        tree_params.run_in_parallel = run_in_parallel
        if not run_in_parallel:
            num_workers = -1
        if num_workers is None:
            num_workers = -1  # this uses all available cpu cores
        tree_params.num_workers = num_workers
        tree_params.only_singleton_splits = only_singleton_splits
        tree_params.max_features = max_features
        self.tree_params = tree_params

        # Args to pass to leaf model's init() function
        self.leafargs = leafargs
        self.leafkwargs = leafkwargs
        self.leafargs_fit = None  # will be updated in following function
        self.leafkwargs_fit = None  # will be updated in following function

        # solver and pred_mode
        self.solver = leafkwargs['solver']
        self.pred_mode = leafkwargs['pred_mode']

        if self.pred_mode == 'SPO-S':
            self.or_ins_num = self.solver.get_num_decisions()
        else:
            # todo remove this stupid var
            self.or_ins_num = 1

        self.solution_dict = {}

    '''
    This function fits the tree on data (X,A,Y,weights).
    
    X: The individual-specific feature data used in tree splits. Can either be a pandas data frame or numpy array, with:
      (a) rows of X = observations/customers
      (b) columns of X = features about the observation/customer
    A: the decision variables/alternative-specific features used in the leaf node models.
      A can take any form -- it is directly passed to the functions in leaf_model.py
    Y: the responses/outcomes/choices used in the leaf node models.
      Y can take any form -- it is directly passed to the functions in leaf_model.py
    D: the pre-calculated decision given the input C
    weights: a numpy array of case weights. Is 1-dimensional, with weights[i] yielding weight of observation/customer i
    feats_continuous: If False, all feature are treated as categorical. If True, all feature are treated as continuous.
      feats_continuous can also be a boolean vector of dimension = num_features specifying how to treat each feature
    verbose: if verbose=True, prints out progress in tree fitting procedure
    refit_leaves: do we refit the leaf models after the best split is found? 
    
    Any additional arguments are passed to the leaf_model fit() function 
    For leaf_model_mnl, you should pass the following:
        n_features:  integer (default is 2)
        mode : "mnl" or "exponomial" (default is "mnl")
        batch_size : size of the stochastic batch (default is 50,)
        model_type : whether the model has alternative varying coefficients or not (default is 0 meaning each alternative has a separate coeff)
        num_features : number of features under consideration (default is 2) 
        epochs : number of epochs for the estimation (default is 10) 
        is_bias : whether the utility function has an intercept (default is True)
    '''

    def fit(self, X, Y,
            weights=None, feats_continuous=False,
            refit_leaves=False, seed=None,
            *leafargs_fit, **leafkwargs_fit):

        # Store the fitted data
        self.fitted_Y = Y
        if self.pred_mode == 'SPO-S':
            p_data_dict = {'D': None,
                           'mu': None,
                           'A': leafkwargs_fit['A'],}
        else:
            p_data_dict = {'D': leafkwargs_fit['D'],
                           'mu': leafkwargs_fit['mu'],
                           'A': leafkwargs_fit['A'],}


        # Set ramdom seed
        if seed is not None:
            np.random.seed(seed)

        # Update params from upper model
        self.leafargs_fit = leafargs_fit
        self.leafkwargs_fit = leafkwargs_fit

        # leaf_params.leafargs_fit = leafargs_fit
        # leaf_params.leafkwargs_fit = leafkwargs_fit

        # The objective value mat A
        A = self.leafkwargs_fit['A']

        # statistic of the data
        num_obs = X.shape[0]
        sub = np.full((X.shape[0]), True).tolist()  # sub: an array conducting inter-layer propagation
        num_features = X.shape[1]  # number of features

        # Store column names if X is a pandas data frame, store the colnames and convert it to a numpy array
        if isinstance(X, pd.core.frame.DataFrame):
            self.Xnames = X.columns.values.tolist()
            X = X.values
        else:
            self.Xnames = ["V" + str(i) for i in range(0, num_features)]
        Xnames = self.Xnames

        # feats_continuous: default setting all continuous var
        if isinstance(feats_continuous, list):
            feats_continuous
        else:
            feats_continuous = [True] * num_features

        # weights
        if weights is None:
            weights = np.ones([num_obs])
        sum_weights = sum(weights) * 1.0  # precompute the sum of weights

        # Gather together leaf parameters
        tree_params = self.tree_params

        # leaf params update
        leaf_params = _LeafParams()
        leaf_params.refit_leaves = refit_leaves
        leaf_params.leafargs = self.leafargs
        leaf_params.leafkwargs = self.leafkwargs

        # Debias_splits: if debias_splits == True,
        # pregenerate the random order of indices for the debias sets
        if tree_params.debias_splits:
            if leafkwargs_fit['verbose']:
                print("Generating debias set")
            # cum_weights = [0] + np.cumsum(weights).astype("int").tolist()
            # weights_alt = np.array([0] * sum_weights.astype("int"))
            # for i in range(0, len(weights)):
            #     weights_alt[cum_weights[i]:cum_weights[i + 1]] = i
            # np.random.shuffle(weights_alt)
            # shuffled_root_inds = weights_alt
            if leafkwargs_fit['verbose']:
                print("Done!")
        else:
            shuffled_root_inds = None

        # Initialize tree
        tree = [_Node()]
        tree[0].set_attr(ind=0,
                         parent_ind=None,
                         depth=0,
                         data_inds=np.array(range(0, num_obs)).tolist())

        # Create the initial leaf and fit the initial data
        leaf_mod = LeafModel(*leaf_params.leafargs, **leaf_params.leafkwargs)
        leaf_mod.fit(Y, A,
                     weights,
                     sub=sub,
                     D=p_data_dict['D'],
                     mu=p_data_dict['mu'],
                     refit=refit_leaves)

        # Calculate the initial loss of tree
        replace_Y = Y.copy()
        if leaf_mod.mean is not None:
            replace_Y = leaf_mod.mean  # we don't have split yet so the whole data will be replaced by avgs
        # todo update this error on first node
        if self.pred_mode == 'SPO-S':
            leaf_mod_error = self.error(A, replace_Y)
        elif self.pred_mode == 'IPTB':
            leaf_mod_error = np.dot(leaf_mod.error(), weights) / sum_weights
        else:
            leaf_mod_error = np.dot(leaf_mod.error(A, replace_Y), weights) / sum_weights

        # update the tree's leaf node 0
        tree[0].set_attr(fitted_model=leaf_mod, fitted_model_error=leaf_mod_error)

        # Initialize the depth nodes vars
        cur_depth_nodes_inds = [0]
        next_depth_nodes_inds = []

        # Start loop: calculating the tree by depth first rule
        while len(cur_depth_nodes_inds) > 0:
            # disable parallel model todo enable
            if tree_params.run_in_parallel:
                # classify each node for parallel computing or single thread computing
                split_node_in_parallel = [_is_node_large(tree[n], X) for n in cur_depth_nodes_inds]
                parallel_nodes_inds = np.array(cur_depth_nodes_inds)[which(split_node_in_parallel)].tolist()
                nonparallel_nodes_inds \
                    = np.array(cur_depth_nodes_inds)[which(np.logical_not(split_node_in_parallel))].tolist()
                cur_depth_child_nodes = np.array([None] * len(cur_depth_nodes_inds))

                if leafkwargs_fit['verbose']:
                    print("Splitting on these nodes in parallel: " + str(parallel_nodes_inds))

                num_workers = min(tree_params.num_workers, len(parallel_nodes_inds))
                num_workers = max(num_workers, 1)  # Should be lager than 1

                cur_depth_child_nodes[which(split_node_in_parallel)] \
                    = Parallel(n_jobs=tree_params.num_workers, max_nbytes=1e5) \
                    (delayed(self._find_best_split)(tree[n], tree_params, leaf_params,
                                                    X, A, Y, weights,
                                                    sum_weights,
                                                    feats_continuous, Xnames,
                                                    shuffled_root_inds)
                     for n in parallel_nodes_inds)

                if leafkwargs_fit['verbose']:
                    print("Splitting on these nodes individually: " + str(nonparallel_nodes_inds))

                cur_depth_child_nodes[which(np.logical_not(split_node_in_parallel))] \
                    = [self._find_best_split(tree[n], tree_params, leaf_params,
                                             X, A, Y, weights,
                                             sum_weights,
                                             feats_continuous, Xnames,
                                             shuffled_root_inds)
                       for n in nonparallel_nodes_inds]
                cur_depth_child_nodes = cur_depth_child_nodes.tolist()
            else:
                cur_depth_child_nodes = [
                    self._find_best_split(tree[n],
                                          tree_params, leaf_params,
                                          X, A, Y, weights,
                                          sum_weights,
                                          feats_continuous, Xnames,
                                          shuffled_root_inds)
                    for n in cur_depth_nodes_inds]

            # For each node in cur_depth_nodes_inds, loop its index to update its attrs
            for i in range(0, len(cur_depth_nodes_inds)):
                n = cur_depth_nodes_inds[i]  # child node id
                tree[n] = cur_depth_child_nodes[i][0]  # parent node model
                child_nodes = cur_depth_child_nodes[i][1]  # child node models
                if child_nodes is not None:

                    # Assign child_nodes indices len(tree),...,len(tree)+len(child_nodes) - 1
                    child_node_inds = range(len(tree), len(tree) + len(child_nodes))

                    # set the ids of the child nodes; have the parent point to the children and vice-versa
                    for j in range(0, len(child_nodes)):
                        child_nodes[j].set_attr(ind=child_node_inds[j],
                                                parent_ind=n,
                                                depth=tree[n].depth + 1)

                    tree[n].set_attr(child_ind=child_node_inds)
                    '''
                    For multi-way splits, set child2val and val2child. Note that the _find_best_split() method sets
                    val2child[val] equal to a "child id", i.e. a number from 0-(#children-1).
                    We convert this to the true child node number in child_node_inds
                    '''
                    if not tree_params.binary_splits:
                        val2child = {}
                        child2val = {}
                        for v, ch_id in tree[n].val2child.items():
                            ch = child_node_inds[ch_id]
                            val2child[v] = ch
                            child2val[ch] = v
                        tree[n].set_attr(val2child=val2child, child2val=child2val)

                    #  Add the child nodes to the tree
                    tree.extend(child_nodes)
                    #  Add the child node indices to the update queue
                    next_depth_nodes_inds.extend(child_node_inds)
                # elif tree[n].is_leaf and self.pred_mode == 'IPTB':
                #     # if the node have no child, means that this node is a leaf:
                #     # get the data inds of node tree[n]
                #     if leafkwargs_fit['verbose']:
                #         print('Starting getting the SAA solution for leaf node ', n, ', having',
                #               sum(data_inds_leaf), 'data samples within this node.')
                #     data_inds_leaf = tree[n].fitted_model.sub
                #     saa_d = self.solver.solve_multiple_models(Y[data_inds_leaf, :],
                #                                               PDO_flag=True)['weights']
                #     tree[n].fitted_model.decision = saa_d
                #     tree[n].saa_optimised = True


            cur_depth_nodes_inds = next_depth_nodes_inds
            next_depth_nodes_inds = []

        # Store tree in self object
        self.tree = tree
        if leafkwargs_fit['verbose']:
            print('the number of leaf nodes are: ', len(self.tree))
            for i in range(0, len(self.tree)):
                if self.tree[i].is_leaf:
                    print('node ', i, ' is leaf: ', self.tree[i].is_leaf)
                    print('node ', i, ' have data size : ', sum(self.tree[i].fitted_model.sub))
                    print('node ', i, ' have depth : ', self.tree[i].depth)

        self._initialize_pruning(leafkwargs_fit['verbose'])

        return

    def error(self, A, Y, sub=None, not_sub=None):
        """
        This is the error function of MTP for SPO-S only
        """

        # Get basic variables
        reshaped_c = self.solver.reshape_y(Y)  # reshape the input y
        n_ins = reshaped_c.shape[0]
        start = time.time()
        # calculate each SPO loss
        SPO_loss_mat = np.zeros([n_ins])
        count = 0

        # loop each instance
        for i in range(n_ins):
            c = reshaped_c[i, :]

            # Look for the str(c) in self.solution_dict.keys()
            if str(c.tolist()) in self.solution_dict.keys():
                decision = self.solution_dict[str(c.tolist())].copy()
            else:
                count += 1
                decision = self.solver.solve_multiple_models(c.reshape([1, -1]))['weights'][0]
                self.solution_dict[str(c.tolist())] = decision
            SPO_loss_mat[i] = self.solver.get_obj(c, decision) - A[i]

        verbose = False
        if verbose:
            print(time.time() - start, count)

        # total SPO loss
        SPO_loss = sum(SPO_loss_mat) / n_ins

        # use sub to divide different split of data.
        if sub is not None:
            l_ids = get_sub(sub, A=np.array(range(n_ins)), Y=Y,
                            or_ins_num=self.or_ins_num,
                            is_boolvec=True)[0]
            r_ids = get_sub(not_sub, A=np.array(range(n_ins)), Y=Y,
                            or_ins_num=self.or_ins_num,
                            is_boolvec=True)[0]

            r_loss = 0
            l_loss = 0

            for i in range(n_ins):
                if i in l_ids and i in r_ids:
                    r_loss += 1 / 2 * SPO_loss_mat[i]
                    l_loss += 1 / 2 * SPO_loss_mat[i]
                elif i in l_ids:
                    l_loss += SPO_loss_mat[i]
                elif i in r_ids:
                    r_loss += SPO_loss_mat[i]
            return SPO_loss, l_loss/n_ins, r_loss/n_ins
        else:
            return SPO_loss

    """
    Performs a split on the given parent node, "node"
    If a split is performed:
      Outputs [node, child nodes] of the split
        Child nodes have the following attributes filled out:
          data_inds, fitted_model, fitted_model_error     
      Sets node.is_leaf=False
      Sets node.split_var_ind, node.is_split_var_numeric,node.split_val
    Else:
      Outputs [node, None]
      Sets node.is_leaf=True
    """

    def _find_best_split(self, node,
                         tree_params,
                         leaf_params,
                         X, A, Y, weights,
                         sum_weights,
                         feats_continuous,
                         Xnames,
                         shuffled_root_inds):
        if tree_params.binary_splits:
            return self._find_best_split_binary(node, tree_params, leaf_params,
                                                X, A, Y, weights,
                                                sum_weights,
                                                feats_continuous, Xnames,
                                                shuffled_root_inds)
        # else:
        #     return _find_best_split_multiway(node, tree_params, leaf_params,
        #                                      X, A, Y, weights,
        #                                      sum_weights, feats_continuous, Xnames, shuffled_root_inds)

    def _find_best_split_binary(self, node,
                                tree_params, leaf_params,
                                X, A, Y,
                                weights, sum_weights,
                                feats_continuous, Xnames,
                                shuffled_root_inds):
        """
        Performs a binary split
        """
        if self.leafkwargs_fit['verbose']:
            print("###############Splitting on node: " + str(node.ind) +
                  ", Depth: " + str(node.depth) +
                  ", Num obs: " + str(len(node.data_inds)))

        # 0. Check if tree is a leaf node(stop splitting), i.e. when
        # (1) max depth has been reached, or
        # (2) minimum weights per node has been reached, or
        # (3) node consists of only data points from one class
        if (node.depth == tree_params.max_depth
                or sum(np.asarray(weights[node.data_inds])) <= tree_params.min_weights_per_node
                or not are_Ys_diverse(get_sub(node.data_inds, A=None, Y=Y, is_boolvec=False))):
            node.is_leaf = True
            del node.data_inds  # we don't need this anymore, so delete for memory management
            return [node, None]  # return leaf node id and no child for None

        """
        1.1 Initialize lists which store split data for each feature
        """

        num_features = X.shape[1]
        # thee averaged error of the split
        # (with binary splits, equals error for best split)
        split_avg_errors = np.array([None] * num_features)
        # split_l_avg_errors[i] yields the left-child-node min avg error corresponding to the
        # optimal split point for variable i
        split_l_avg_errors = np.array([None] * num_features)
        # split_r_avg_errors[i] yields the right-child-node min avg error corresponding to the
        # optimal split point for variable i
        split_r_avg_errors = np.array([None] * num_features)
        # split_l_fitted_model[i] yields the left-child-node model corresponding to the
        # optimal split point for variable i
        split_l_fitted_model = [None] * num_features
        # split_r_fitted_model[i] yields the right-child-node model corresponding to the
        # optimal split point for variable i
        split_r_fitted_model = [None] * num_features
        # split_vals[i] yields the best split value when splitting on variable i
        split_vals = [None] * num_features

        # create debias set if appropriate
        debias_set_size = np.floor(sum(np.asarray(weights[node.data_inds])) * tree_params.frac_debias_set).astype(int)

        # only debias splits if debias_set_size meets the minimum size requirement
        debias_splits = tree_params.debias_splits and (debias_set_size >= tree_params.min_debias_set_size)

        '''
        1.2 Debias codes
        '''

        if not debias_splits:
            weights_train = np.asarray(weights[node.data_inds])
            data_inds_train = node.data_inds
        else:
            raise ValueError('debias currently disabled')
        """
        else:
            '''
            Divide data (represented by weight vector "weights") into training set
            and debias set.
             Equivalent: create two vectors weights_train and weights_debias
            such that weights[node.data_inds] = weights_train[node.data_inds]+weights_debias[node.data_inds]
            '''
            if (verbose == True): print("Node " + str(node.ind) + ": Generating Debias Set")
            # extract elements of shuffled_root_inds that are equal to data_inds, maintaining random
            # order of indices
            tmp = np.in1d(shuffled_root_inds, node.data_inds)
            shuffled_node_inds = np.asarray(shuffled_root_inds[tmp])
            # take first debias_set_size entries of shuffled_node_inds to make debias set
            shuffled_node_inds_debias = shuffled_node_inds[:debias_set_size]
            # use the rest of the data for training
            shuffled_node_inds_train = shuffled_node_inds[debias_set_size:]
            # create weights_debias and weights_train sets
            weights_debias = np.zeros(len(weights))
            weights_train = np.zeros(len(weights))
            inds_debias, counts_debias = np.unique(shuffled_node_inds_debias, return_counts=True)
            inds_train, counts_train = np.unique(shuffled_node_inds_train, return_counts=True)
            weights_debias[inds_debias.tolist()] = counts_debias
            weights_train[inds_train.tolist()] = counts_train
            # counts_debias = shuffled_node_inds_debias.value_counts()
            # counts_train = shuffled_node_inds_train.value_counts()
            # weights_debias[counts_debias.index.tolist()] = counts_debias
            # weights_train[counts_train.index.tolist()] = counts_train
            data_inds_debias = np.array(node.data_inds)[weights_debias[node.data_inds] != 0].tolist()
            data_inds_train = np.array(node.data_inds)[weights_train[node.data_inds] != 0].tolist()
        """

        '''1.3 Prepare training data and input params in this node '''
        X_node = X[data_inds_train, :]
        weights_node = weights[data_inds_train]

        # Picking the features for
        # Only consider a random subset of "max_features" features to split on
        if tree_params.max_features == "all":
            splitting_x_inds = range(0, num_features)
        elif tree_params.max_features == "auto":
            # using the root num of num_features for feature selection
            splitting_x_inds = np.sort(
                np.random.choice(range(0, num_features), int(np.sqrt(num_features)), replace=False))
        else:
            # use num_features
            splitting_x_inds = np.sort(
                np.random.choice(range(0, num_features), tree_params.max_features, replace=False))

        '''2. For each covariate j, calculate the best possible error splitting on j'''
        for j in splitting_x_inds:
            if self.leafkwargs_fit['verbose']:
                print("Node " + str(node.ind) + ": Splitting on covariate " + Xnames[j])
            Xj = np.asarray(X_node[:, j])
            # note: np.unique() sorts the unique values
            uniq_Xj = np.unique(Xj)
            # if there only exists one feature value, we cannot split on this covariate
            if len(uniq_Xj) <= 1:
                continue

            """
            2.2 find the candidate split points for continuous and discontinuous feats
            Two method are applied:
            1. if quant_discret was given within (0,0.5), use quantiles to find the split points   
            2. otherwise, use a midpoint method, which only have a half split each time
            """
            # Assuming uniq_Xj is already sorted and contains unique values of the continuous feature
            if tree_params.quant_discret <= 0 or tree_params.quant_discret >= 0.5:
                if feats_continuous[j]:
                    # If there are enough unique values, compute the midpoints
                    if len(uniq_Xj) > 1:
                        # Calculate midpoints as the average of consecutive unique values
                        candidate_split_vals = (uniq_Xj[:-1] + uniq_Xj[1:]) / 2.0
                    else:
                        # If there's only one unique value, no split is possible
                        candidate_split_vals = []
                else:
                    # For categorical features, use all unique values as candidate split points
                    candidate_split_vals = uniq_Xj
                    # Optimization for binary categorical features
                    if len(candidate_split_vals) == 2:
                        candidate_split_vals = [candidate_split_vals[0]]
            else:
                if feats_continuous[j]:
                    # splits are at quantiles tree_params.quant_discret, ..., n*tree_params.quant_discret
                    num_splits = int(np.ceil(1.0 / tree_params.quant_discret)) - 1
                    if num_splits < len(uniq_Xj) - 1:
                        quants = [tree_params.quant_discret * i * 100
                                  for i in range(1, num_splits + 1)]
                        candidate_split_vals = np.percentile(uniq_Xj,
                                                             quants,
                                                             interpolation="higher")
                    else:
                        # Simply test all unique values
                        candidate_split_vals = uniq_Xj[1:len(uniq_Xj)]
                else:
                    # use all unique values of the covariate observed in the training set as the candidate split points
                    # note: we will actually split on SUBSETS of candidate_split_vals
                    candidate_split_vals = uniq_Xj
                    # if there exists two categorical features, we only have to test splitting on
                    # one of them
                    if len(candidate_split_vals) == 2:
                        candidate_split_vals = [candidate_split_vals[0]]

            '''2.3 try all candidate split points (continuous values)'''

            if feats_continuous[j]:
                if self.leafkwargs_fit['verbose']:
                    print("Node " + str(node.ind) + ": Finding Best < Split")
                num_splits = len(candidate_split_vals)
            else:
                if tree_params.only_singleton_splits:
                    if self.leafkwargs_fit['verbose']:
                        print("Node " + str(node.ind) + ": Finding Best Singleton Split")
                    num_splits = 1
                else:
                    if self.leafkwargs_fit['verbose']:
                        print("Node " + str(node.ind) + ": Finding Best Subset Split")
                    num_splits = max(1, np.floor(len(candidate_split_vals) / 2.0).astype("int"))

            # Initialize vars.
            l_avg_errors = np.array([None] * num_splits)
            # r_avg_errors[k] = best errors with subset of size k corresponding to right child node
            r_avg_errors = np.array([None] * num_splits)
            # Note: avg_errors = l_avg_errors + r_avg_errors
            avg_errors = np.array([None] * num_splits)

            # store the fitted model for the best subsets of size k, for all k
            l_fitted_model = [None] * num_splits
            r_fitted_model = [None] * num_splits

            if feats_continuous[j]:
                # Loop of 2.3
                split_vals_j = candidate_split_vals
            else:
                # Store the split corresponding to the best subset of size k, for all k
                split_vals_j = [None] * num_splits

            for k in range(0, num_splits):
                # fit_init_l and r are for storing the fitting node model to right and left split (for calculation)
                if k == 0:
                    fit_init_l = node.fitted_model
                    fit_init_r = node.fitted_model
                    if not feats_continuous[j]:
                        base_sub = np.array([])  # base_sub = best subset of size k-1
                else:
                    if not feats_continuous[j]:
                        base_sub = split_vals_j[k - 1]  # for continues vars

                    # Initial left if input is None
                    if l_fitted_model[k - 1] is None:
                        fit_init_l = node.fitted_model
                    else:
                        # Normal initial
                        fit_init_l = l_fitted_model[k - 1]
                    # Initial right
                    if r_fitted_model[k - 1] is None:
                        fit_init_r = node.fitted_model
                    else:  # Normal initial
                        fit_init_r = r_fitted_model[k - 1]

                if not feats_continuous[j] and not tree_params.only_singleton_splits:
                    # Possible split values to try for best subset of size k
                    addl_split_vals = np.setdiff1d(candidate_split_vals, base_sub)
                    l_avg_errors_k = np.array([None] * len(addl_split_vals))
                    r_avg_errors_k = np.array([None] * len(addl_split_vals))
                    avg_errors_k = np.array([None] * len(addl_split_vals))
                    l_fitted_model_k = [None] * len(addl_split_vals)
                    r_fitted_model_k = [None] * len(addl_split_vals)

                    ######################################################
                    splits = [None] * len(addl_split_vals)
                    for l in range(0, len(addl_split_vals)):
                        sub = np.concatenate([base_sub, [addl_split_vals[l]]])
                        # get the boolean array of left or right split
                        tmp = np.full((Xj.shape[0]), False).tolist()
                        not_tmp = np.full((Xj.shape[0]), False).tolist()
                        for i in node.data_inds:
                            if Xj[i] in sub:
                                tmp[i] = True
                            else:
                                not_tmp[i] = True
                        splits[l] = self._perform_splitting(node,
                                                            tmp, not_tmp, A, Y,
                                                            weights_train, sum_weights,
                                                            leaf_params,
                                                            fit_init_l, fit_init_r)
                    # for each split, update valuex
                    for l in range(0, len(splits)):
                        l_data, r_data = splits[l]

                        # print splits[l]
                        l_fitted_model_k[l] = l_data[0]
                        r_fitted_model_k[l] = r_data[0]
                        l_avg_errors_k[l] = l_data[1]
                        r_avg_errors_k[l] = r_data[1]
                        avg_errors_k[l] = l_avg_errors_k[l] + r_avg_errors_k[l]

                    min_ind = np.argmin(avg_errors_k)

                    split_vals_j[k] = np.concatenate([base_sub, [addl_split_vals[min_ind]]])
                    # would performing this split result in one of the tree conditions (e.g., min wts per node)
                    # not being satisfied?
                    is_split_legal_k = _is_split_legal(split_vals_j[k], Xj, tree_params.min_weights_per_node,
                                                       weights_node)

                    if is_split_legal_k and (avg_errors_k[min_ind] != float("inf")):
                        avg_errors[k] = avg_errors_k[min_ind]
                        l_avg_errors[k] = l_avg_errors_k[min_ind]
                        r_avg_errors[k] = r_avg_errors_k[min_ind]
                        l_fitted_model[k] = l_fitted_model_k[min_ind]
                        r_fitted_model[k] = r_fitted_model_k[min_ind]
                    else:
                        avg_errors[k] = None

                else:
                    if feats_continuous[j]:
                        tmp = []
                        not_tmp = []
                        for i in range(len(node.data_inds)):
                            if X[node.data_inds[i], j] < candidate_split_vals[k]:
                                tmp.append(i)
                            else:
                                not_tmp.append(i)
                    else:
                        tmp = []
                        not_tmp = []
                        for i in range(len(node.data_inds)):
                            if X[node.data_inds[i], j] == candidate_split_vals[k]:
                                tmp.append(i)
                            else:
                                not_tmp.append(i)

                    # get splitting list
                    l_list, r_list = self._perform_splitting(node,
                                                             tmp, not_tmp,
                                                             weights_train, sum_weights,
                                                             leaf_params,
                                                             fit_init_l, fit_init_r)


                    if (r_list[1] == 1) or (l_list[1] == 1):
                        # if max error, skip this leaf
                        continue
                    else:
                        l_fitted_model[k] = l_list[0]
                        r_fitted_model[k] = r_list[0]

                        # find the right (not tmp nodes)
                        A_l, Y_l = get_sub(tmp, or_ins_num=self.or_ins_num, A=A, Y=Y, is_boolvec=True)
                        A_r, Y_r = get_sub(not_tmp, or_ins_num=self.or_ins_num, A=A, Y=Y, is_boolvec=True)

                        weights_train_l = weights_train[tmp]
                        weights_train_r = weights_train[not_tmp]

                        if self.or_ins_num == 1:
                            if self.pred_mode == 'SPO-M':
                                l_avg_errors[k] = np.dot(l_fitted_model[k].error(A_l, Y_l), weights_train_l) / sum_weights
                                r_avg_errors[k] = np.dot(r_fitted_model[k].error(A_r, Y_r), weights_train_r) / sum_weights
                            elif self.pred_mode == 'IPTB':
                                l_avg_errors[k] = np.dot(l_fitted_model[k].error(), weights_train_l) / sum_weights
                                r_avg_errors[k] = np.dot(r_fitted_model[k].error(), weights_train_r) / sum_weights
                            avg_errors[k] = l_avg_errors[k] + r_avg_errors[k]
                        else:
                            # use replaced_Y to calculate the errors
                            replace_Y = Y.copy()
                            replace_Y[tmp] = l_fitted_model[k].mean
                            replace_Y[not_tmp] = r_fitted_model[k].mean
                            avg_errors[k], l_avg_errors[k], r_avg_errors[k] = self.error(A, replace_Y,
                                                                                         sub=tmp, not_sub=not_tmp)

                        if self.leafkwargs_fit['verbose']:
                            print(str(k) + "th split splits on" + str(candidate_split_vals[k])
                                  + ": avg. error is " + str(avg_errors[k]))

            # 3. splitval_candidates = incides corresponding to elements which are not None in vector avg_errors
            splitval_candidates = which([x is not None for x in avg_errors])
            if len(splitval_candidates) == 0:
                # no candidate, make this node as leaf
                split_avg_errors[j] = None
            else:
                # min_ind = index of minimum value in avg_errors that is not None
                min_ind = splitval_candidates[np.argmin(avg_errors[splitval_candidates])]
                split_avg_errors[j] = avg_errors[min_ind]
                split_l_avg_errors[j] = l_avg_errors[min_ind]
                split_r_avg_errors[j] = r_avg_errors[min_ind]
                split_vals[j] = split_vals_j[min_ind]
                split_l_fitted_model[j] = l_fitted_model[min_ind]
                split_r_fitted_model[j] = r_fitted_model[min_ind]

        split_candidates = which([x is not None for x in split_avg_errors])
        # if there exists no variables left to split on, then make "n" a leaf node
        if len(split_candidates) == 0:
            node.is_leaf = True
            del node.data_inds  # we don't need this anymore, so delete for memory management
            return [node, None]

        ####################
        """
        # If applicable, debias splits, i.e. recompute model errors on held-out set
        if (debias_splits == True):

            # Prepare training data in this node
            X_node = X[data_inds_debias, :]
            A_node, Y_node = get_sub(data_inds_debias, A=A, Y=Y, or_ins_num=self.or_ins_num, is_boolvec=False)
            weights_node = weights[data_inds_debias]
            weights_debias_node = weights_debias[data_inds_debias]

            for j in split_candidates:
                Xj = np.asarray(X_node[:, j]);
                if feats_continuous[j] == True:
                    tmp = (Xj < split_vals[j])
                else:
                    tmp = np.in1d(Xj, split_vals[j])
                not_tmp = np.logical_not(tmp)
                A_l, Y_l = get_sub(tmp, A=A_node, Y=Y_node, or_ins_num=self.or_ins_num, is_boolvec=True)
                A_r, Y_r = get_sub(not_tmp, A=A_node, Y=Y_node, or_ins_num=self.or_ins_num, is_boolvec=True)
                weights_debias_l = weights_debias_node[tmp]
                weights_debias_r = weights_debias_node[not_tmp]
                if (sum(weights_debias_l) == 0):
                    split_l_avg_errors[j] = 0;
                else:
                    split_l_avg_errors[j] = np.dot(split_l_fitted_model[j].error(A_l, Y_l),
                                                   weights_debias_l) / sum_weights;

                if (sum(weights_debias_r) == 0):
                    split_r_avg_errors[j] = 0;
                else:
                    split_r_avg_errors[j] = np.dot(split_r_fitted_model[j].error(A_r, Y_r),
                                                   weights_debias_r) / sum_weights;

                split_avg_errors[j] = split_l_avg_errors[j] + split_r_avg_errors[j]
        """
        ####################

        # if the best possible split produces infinite error, then make this a leaf node
        if min(split_avg_errors[split_candidates]) == float("inf"):
            node.is_leaf = True
            del node.data_inds  # we don't need this anymore, so delete for memory management
            return [node, None]


        # 4. Choose the best split
        # use min error split
        node.split_var_ind = split_candidates[np.argmin(split_avg_errors[split_candidates])]
        node.is_split_var_numeric = feats_continuous[node.split_var_ind]
        node.split_val = split_vals[node.split_var_ind]

        # create child nodes, first performing the chosen split
        # get data_inds for left and right child nodes
        l_split_data_inds, r_split_data_inds, tmp, not_tmp \
            = get_data_inds(X, node.data_inds, node.split_var_ind, node.split_val, feats_continuous)

        l_node = _Node()
        r_node = _Node()
        ##################################################################################
        ##################################################################################
        ##################################################################################
        # if debias = True, retrain on debias + training set, rewrite error and fitted model
        # debias mode is set off todo: recode debias mode

        if debias_splits or leaf_params.refit_leaves:
            A_l, Y_l = get_sub(l_split_data_inds, A=A, Y=Y, or_ins_num=self.or_ins_num, is_boolvec=False)
            A_r, Y_r = get_sub(r_split_data_inds, A=A, Y=Y, or_ins_num=self.or_ins_num, is_boolvec=False)
            weights_l = np.asarray(weights[l_split_data_inds])
            weights_r = np.asarray(weights[r_split_data_inds])

            # leaf_mod_l = LeafModel(*self.leafargs,**self.leafkwargs)
            # leaf_mod_r = LeafModel(*self.leafargs,**self.leafkwargs)

            leaf_mod_l = split_l_fitted_model[node.split_var_ind]
            leaf_mod_r = split_r_fitted_model[node.split_var_ind]

            # # .fit() returns 0 or 1 corresponding to whether an error occurred during fitting process
            # error_l = leaf_mod_l.fit(A_l, Y_l, weights_l,
            #                          refit=leaf_params.refit_leaves,
            #                          *leaf_params.leafargs_fit,
            #                          **leaf_params.leafkwargs_fit)
            # error_r = leaf_mod_r.fit(A_r, Y_r, weights_r,
            #                          refit=leaf_params.refit_leaves,
            #                          *leaf_params.leafargs_fit,
            #                          **leaf_params.leafkwargs_fit)
            #
            # if (error_l == 1) or (error_r == 1):
            #     avg_error = float("inf")  # do not consider this split
            # else:
            #     l_avg_error = np.dot(leaf_mod_l.error(A_l, Y_l), weights_l) / sum_weights
            #     r_avg_error = np.dot(leaf_mod_r.error(A_r, Y_r), weights_r) / sum_weights
            #
            #     avg_error = l_avg_error + r_avg_error

            # if avg_error == float("inf"):
            #     node.is_leaf = True
            #     del node.data_inds  # we don't need this anymore, so delete for memory management
            #     return [node, None]
            l_avg_error = np.dot(leaf_mod_l.error(A_l, Y_l), weights_l) / sum_weights
            r_avg_error = np.dot(leaf_mod_r.error(A_r, Y_r), weights_r) / sum_weights

            avg_error = l_avg_error + r_avg_error
            l_node.set_attr(data_inds=l_split_data_inds, fitted_model=leaf_mod_l, fitted_model_error=l_avg_error)
            r_node.set_attr(data_inds=r_split_data_inds, fitted_model=leaf_mod_r, fitted_model_error=r_avg_error)
            ##################################################################################
            ##################################################################################
            ##################################################################################
        else:
            l_node.set_attr(data_inds=l_split_data_inds,
                            fitted_model=split_l_fitted_model[node.split_var_ind],
                            fitted_model_error=split_l_avg_errors[node.split_var_ind])
            r_node.set_attr(data_inds=r_split_data_inds,
                            fitted_model=split_r_fitted_model[node.split_var_ind],
                            fitted_model_error=split_r_avg_errors[node.split_var_ind])

        # TODO ADD CHECK STOP CRITERIA
        # if we have exceeded the minimum depth, AND
        # performing such a split would lead to a HIGHER overall error than not splitting (w.r.t. min_diff),
        # then make this a leaf node
        if node.depth >= tree_params.min_depth:
            if ((l_node.fitted_model_error + r_node.fitted_model_error) - (
                    -1.0 * node.fitted_model_error) >= -1.0 * tree_params.min_diff):
                node.is_leaf = True
                del node.data_inds  # we don't need this anymore, so delete for memory management
                return [node, None]

        node.is_leaf = False
        del node.data_inds  # we don't need this anymore, so delete for memory management
        return [node, [l_node, r_node]]

    def _perform_splitting(self, node,
                           tmp, not_tmp,
                           weights_train, sum_weights,
                           leaf_params, fit_init_l, fit_init_r):
        """
        Tool function for binary split or multiway spliting
        """

        # Get the data according to tmp and not_tmp
        # get A and Y
        A_l, Y_l = get_sub(tmp,
                           or_ins_num=self.or_ins_num,
                           A=node.fitted_model.A, Y=node.fitted_model.Y,
                           is_boolvec=True)
        A_r, Y_r = get_sub(not_tmp,
                           or_ins_num=self.or_ins_num,
                           A=node.fitted_model.A, Y=node.fitted_model.Y,
                           is_boolvec=True)

        # get the D and mu if in IPTB
        if self.pred_mode == 'IPTB':
            D_l = node.fitted_model.D[tmp]
            D_r = node.fitted_model.D[not_tmp]
            mu_l = node.fitted_model.mu[tmp]
            mu_l = mu_l[:, tmp]
            mu_r = node.fitted_model.mu[not_tmp]
            mu_r = mu_r[:, not_tmp]

        # get the weights of the data
        weights_train_l = weights_train[tmp]
        weights_train_r = weights_train[not_tmp]

        leaf_mod_l = LeafModel(*leaf_params.leafargs, **leaf_params.leafkwargs)
        leaf_mod_r = LeafModel(*leaf_params.leafargs, **leaf_params.leafkwargs)

        # .fit() returns 0 or 1 corresponding to whether an error occurred during fitting process
        if self.pred_mode == 'IPTB':
            error_l = leaf_mod_l.fit(Y_l, A_l, weights_train, sub=tmp,
                                     D=D_l, mu=mu_l,
                                     fit_init=fit_init_l)
            error_r = leaf_mod_r.fit(Y_r, A_r, weights_train, sub=not_tmp,
                                     fit_init=fit_init_r,
                                     D=D_r, mu=mu_r)
        else:
            error_l = leaf_mod_l.fit(Y_l, A_l, weights_train_l,
                                     fit_init=fit_init_l)
            error_r = leaf_mod_r.fit(Y_r, A_r, weights_train_r,
                                     fit_init=fit_init_r)

        # if any error is 1
        if (error_l == 1) or (error_r == 1):
            # do not consider this split
            l_fitted_model = None
            r_fitted_model = None
            l_avg_errors = np.float("inf")
            r_avg_errors = np.float("inf")
        else:
            l_fitted_model = leaf_mod_l
            r_fitted_model = leaf_mod_r

            if self.or_ins_num == 1:
                if self.pred_mode == 'SPO-M':
                    l_avg_errors = np.dot(leaf_mod_l.error(A_l, Y_l), weights_train_l) / sum_weights
                    r_avg_errors = np.dot(leaf_mod_r.error(A_r, Y_r), weights_train_r) / sum_weights
                elif self.pred_mode == 'IPTB':
                    l_avg_errors = np.dot(leaf_mod_l.error(), weights_train_l) / sum_weights
                    r_avg_errors = np.dot(leaf_mod_r.error(), weights_train_r) / sum_weights
            else:
                # get replaced_Y
                replace_Y = node.fit_model.Y.copy()
                replace_Y[tmp] = leaf_mod_l.mean
                replace_Y[not_tmp] = leaf_mod_r.mean
                avg_errors, l_avg_errors, r_avg_errors \
                    = self.error(node.fit_model.A, replace_Y, sub=tmp, not_sub=not_tmp)

        return [[l_fitted_model, l_avg_errors], [r_fitted_model, r_avg_errors]]


    # returns encoding of tree a,b, where a = split variables of tree, b = split points of tree
    # optional: if x_train is specified, output mapping of train obs to leaves as well
    def get_tree_encoding(self, x_train=None):
        num_nonterminal_nodes = 2 ** self.tree_params.max_depth - 1
        num_leaves = 2 ** self.tree_params.max_depth
        num_features = len(self.Xnames)
        a = np.zeros((num_features, num_nonterminal_nodes), dtype=int)
        b = np.ones(num_nonterminal_nodes)
        CART2TRANSFORMinternal_node_ids = {0: 0}

        nodes_to_traverse = [0]
        while (len(nodes_to_traverse) > 0):
            n = nodes_to_traverse.pop(0)
            n_transform = CART2TRANSFORMinternal_node_ids[n]

            if self.alpha_best >= self.tree[n].alpha_thresh or self.tree[n].is_leaf:
                continue
            else:
                assert (self.tree_params.binary_splits == True)
                assert (self.tree[n].is_split_var_numeric == True)
                a[self.tree[n].split_var_ind, n_transform] = 1
                b[n_transform] = self.tree[n].split_val
                CART2TRANSFORMinternal_node_ids[self.tree[n].child_ind[0]] = 2 * n_transform + 1
                CART2TRANSFORMinternal_node_ids[self.tree[n].child_ind[1]] = 2 * n_transform + 2
                nodes_to_traverse.extend(self.tree[n].child_ind)

        if x_train is None:
            return a, b
        else:
            def decision_path(x, a, b):
                T_B = len(b)
                if len(x.shape) == 1:
                    n = 1
                    P = x.size
                else:
                    n, P = x.shape
                res = []
                for i in range(n):
                    node = 0
                    path = [0]
                    T_B = a.shape[1]
                    while node < T_B:
                        if np.dot(x[i, :], a[:, node]) < b[node]:
                            node = (node + 1) * 2 - 1
                        else:
                            node = (node + 1) * 2
                        path.append(node)
                    res.append(path)
                return np.array(res)

            n_train = x_train.shape[0]
            z = np.zeros((n_train, num_leaves), dtype=int)
            path = decision_path(x_train, a, b)
            paths = path[:, -1]
            for i in range(n_train):
                decision_node = paths[i] - num_leaves + 1
                z[i, decision_node] = 1
            return a, b, z

    '''
    Prints out the tree. 
    Required: call tree fit() method first
    Prints pruned tree if prune() method has been called, else prints unpruned tree
    verbose=True prints additional statistics within each leaf
    
    Any additional arguments are passed to the leaf_model to_string() function
    '''

    def traverse(self, *leafargs, **leafkwargs):
        nodes_to_traverse = [0];
        num_leaf_nodes = 0.0;
        num_nodes = 0.0;
        max_depth = 0.0;
        while (len(nodes_to_traverse) > 0):
            n = nodes_to_traverse.pop(0);
            num_nodes = num_nodes + 1.0;
            max_depth = max(max_depth, self.tree[n].depth);
            print("Node " + str(n) + ": Depth " + str(self.tree[n].depth));
            if (self.tree[n].parent_ind is None):
                print("Parent Node: NA <Root>")
            else:
                print("Parent Node: " + str(self.tree[n].parent_ind))

            # if (verbose == True):
            # print("Model error (if this were a leaf node): " + str(self.tree[n].fitted_model_error))
            # print("Alpha Threshold: " + str(self.tree[n].alpha_thresh))

            if ((self.alpha_best >= self.tree[n].alpha_thresh) or (self.tree[n].is_leaf == True)):
                # if self.tree[n].is_leaf == True:
                num_leaf_nodes = num_leaf_nodes + 1.0;
                print("Leaf: " + self.tree[n].fitted_model.to_string(*leafargs, **leafkwargs))
            #        try:
            #            print("Model quality: ",T.tree[n].fitted_model.model_quality)
            #        except:
            #            print("Model quality unavailable")

            else:
                if (self.tree_params.binary_splits == True):
                    if (self.tree[n].is_split_var_numeric):
                        print(
                            "Non-terminal node splitting on (numeric) var " + self.Xnames[self.tree[n].split_var_ind]);
                        print("Splitting Question: Is " + self.Xnames[self.tree[n].split_var_ind] + " < " + str(
                            self.tree[n].split_val) + "?");
                    else:
                        print("Non-terminal Node splitting on (categorical) var " + self.Xnames[
                            self.tree[n].split_var_ind])
                        print("Splitting Question: Is " + self.Xnames[self.tree[n].split_var_ind] + " in: ")
                        if (len(self.tree[n].split_val) <= 100):
                            print(str(self.tree[n].split_val));
                        else:
                            print("Subset of length " + str(len(self.tree[n].split_val)));
                    print("Child Nodes: " + str(self.tree[n].child_ind[0]) + " (True), " + str(
                        self.tree[n].child_ind[1]) + " (False)")
                #          print("Model Quality: ",self.tree[n].fitted_model.model_quality)
                # else we perform multi-way splits
                else:
                    print("Non-terminal node splitting on var " + self.Xnames[self.tree[n].split_var_ind]);
                    child_ind = self.tree[n].child_ind
                    print("Child Nodes: " + str(child_ind[0]) + "-" + str(child_ind[-1]))
                    #          try:
                    #              print("Model quality: ",self.tree[n].fitted_model.model_quality)
                    #          except:
                    #            print("Model quality unavailable")
                    if (len(child_ind) <= 50):
                        str_ch = "Child node (feature value): ";
                        child2val = self.tree[n].child2val
                        for ch in child_ind:
                            str_ch = str_ch + str(ch) + " (" + str(child2val[ch]) + "), "
                        print(str_ch)
                # add new nodes to the queue
                nodes_to_traverse.extend(self.tree[n].child_ind)

            print("\n")

        self.num_leaf_nodes = num_leaf_nodes
        print("Max depth:" + str(max_depth))
        print("Num. Nodes: " + str(num_nodes))
        print("Num. Terminal Nodes: " + str(num_leaf_nodes))
        return

    '''
    Prunes the tree. Set verbose=True to track progress
    approx_pruning: should we use faster pruning method which finds approximately the best alpha?
    '''

    def prune(self, Xval, Aval, Yval, weights_val=None, one_SE_rule=True, verbose=False, approx_pruning=False):

        # If Xval is a pandas data frame, convert it to a numpy array
        if (isinstance(Xval, pd.core.frame.DataFrame)):
            Xval = Xval.values

        num_val_obs = Xval.shape[0]
        # set weights_val
        if weights_val is None:
            weights_val = np.ones([num_val_obs])

        # only do the fast (approximate) pruning procedure when evaluating on more than 100 subtrees
        approx_pruning = (approx_pruning and len(self.alpha_seq) >= 100)

        if approx_pruning == False:
            if verbose == True:
                print("Conducting CART Pruning Method")
            print(self.alpha_seq)
            self.alpha_best, _ = self.prune_find_best_alpha(self.alpha_seq, Xval, Aval, Yval, weights_val, one_SE_rule,
                                                            verbose)
        else:
            K = len(self.alpha_seq)
            num_intervals = int(np.floor(np.sqrt(2 * K) + 1))
            interval_inds = np.unique(np.linspace(0, K - 1, num_intervals).astype(int))
            interval_alpha_seq = self.alpha_seq[interval_inds]
            if verbose == True:
                print("Conducting Approximate CART Pruning Method: Pass 1/2")
            _, i_best = self.prune_find_best_alpha(interval_alpha_seq, Xval, Aval, Yval, weights_val, one_SE_rule,
                                                   verbose)

            if i_best == 0:
                l_ind = 0
                r_ind = interval_inds[i_best + 1] - 1
            elif i_best == len(interval_inds) - 1:
                l_ind = interval_inds[i_best - 1] + 1
                r_ind = interval_inds[-1]
            else:
                l_ind = interval_inds[i_best - 1] + 1
                r_ind = interval_inds[i_best + 1] - 1

            within_interval_alpha_seq = self.alpha_seq[range(l_ind, r_ind + 1)]
            if verbose == True:
                print("Conducting Approximate CART Pruning Method: Pass 2/2")
            self.alpha_best, _ = self.prune_find_best_alpha(within_interval_alpha_seq, Xval, Aval, Yval, weights_val,
                                                            one_SE_rule, verbose)
        return

    '''
    helper method for prune(). finds best alpha in alpha_seq and index corresponding to that alpha
    '''

    def prune_find_best_alpha(self, alpha_seq, Xval, Aval, Yval, weights_val, one_SE_rule, verbose):
        num_val_obs = Xval.shape[0]

        val_error = np.array([None] * len(alpha_seq));
        SE_val_error = np.array([None] * len(alpha_seq));
        # Xloc: location of data Xval in the tree corresponding to the current alpha.
        # See pruning documentation for how Xloc is encoded
        # Pruning starts with alpha=Inf (full pruning), so starting location is root node 0
        XlocNerrors = ([0], [np.array(range(0, num_val_obs))], {})
        for alpha_ind in reversed(range(0, len(alpha_seq))):
            errors, XlocNerrors = self._error(Xval, Aval, Yval, alpha=alpha_seq[alpha_ind], return_locNerrors=True,
                                              init_locNerrors=XlocNerrors, use_pruning_error=True)
            avg_error = fast_avg(errors, weights_val)
            # (weighted) variance of errors
            var_errors = np.dot(weights_val, (errors - avg_error) ** 2.0) / (sum(weights_val) - 1.0)
            # standard error
            se_avg_error = np.sqrt(var_errors) / np.sqrt(sum(weights_val));

            val_error[alpha_ind] = avg_error
            SE_val_error[alpha_ind] = se_avg_error

            if (verbose == True):
                print("Testing subtree " + str(len(alpha_seq) - alpha_ind) + " out of " + str(len(alpha_seq)));

        # val_error[alpha_ind] represents the error from parameter alpha_ind
        # SE_val_error[alpha_ind] represents the standard error for val_error[alpha_ind]

        #####TEST: NOT CHOOSING THE ALPHA == INF'S POS
        # val_error[-1] = np.Inf
        min_ind = np.argmin(val_error);
        if min_ind == len(alpha_seq) - 1:
            print("infinitive alpha is used as min error purning,"
                  " please check the calculation of pruning step.")
        if (one_SE_rule == False):
            tmp = which(val_error <= val_error[min_ind] + abs(val_error[min_ind]) * 1e-5);  # why this?
            alpha_best = max(alpha_seq[tmp])
            return alpha_best, np.where(alpha_seq == alpha_best)[0][0]
            # return alpha_seq[min_ind], min_ind
        else:
            tmp = which(val_error <= val_error[min_ind] + SE_val_error[min_ind]);
            alpha_best_1se = max(alpha_seq[tmp])
            return alpha_best_1se, np.where(alpha_seq == alpha_best_1se)[0][0]

    '''
    def refit(self, X,A,Y,alpha = None, steps = 20000):
  
      if alpha is None:
        alpha = self.alpha_best
        
      num_obs = X.shape[0]
      unq, unq_inds_vec = self._find_leaf_nodes(0,X,np.array(range(0,num_obs)),alpha);
      new_args = self.leafkwargs_fit.copy()
      new_args['steps'] = 20000
      new_args['learning_rate'] = 0.005    
      for i in range(0,len(unq)):
            n = unq[i]
            unq_inds = unq_inds_vec[i]
            self.tree[n].fitted_model.fit(get_sub(unq_inds,A),
                                          get_sub(unq_inds,Y),
                                          weights = np.ones(Y.shape[0]),
                                          *self.leafargs_fit,**new_args)
      return 
    '''

    '''
    Predicts response data given Xnew, Anew
    Required: call tree fit() method first
    Uses pruned tree if pruning method has been called, else uses unpruned tree
    Argument alpha controls level of pruning. If not specified, uses alpha trained from the prune() method
    
    As a step in finding the estimated probabilities for data (Xnew,Anew), this function first finds
    the leaf node locations corresponding to each row of Xnew. It does so by a top-down search
    starting at the root node 0. 
    If return_loc=True, predict() will also return the leaf node locations for the data, in addition to the prob estimates.
    
    Any additional arguments are passed to the leaf_model predict() function
    '''

    def predict(self, Xnew, Anew, alpha=None, return_loc=False, get_cost=False, *leafargs, **leafkwargs):
        """
        Returns the response predicted probabilities for the given data Xnew, Anew
        """

        # If Xnew is a pandas data frame, convert it to a numpy array
        if (isinstance(Xnew,
                       pd.core.frame.DataFrame)):
            Xnew = Xnew.values

        if alpha is None:
            alpha = self.alpha_best

        num_obs = Xnew.shape[0]

        unq, unq_inds_vec = self._find_leaf_nodes(0, Xnew, np.array(range(0, num_obs)), alpha)

        """
        Returns unq,unq_inds_vec:
      - unq: a vector giving the unique leaf node indices corresponding to X
      - unq_inds_vec: unq_inds_vec[i] are X's observation indices in leaf unq[i], i.e. X[unq_inds_vec[i],:] are in leaf unq[i]
        """

        for i in range(0, len(unq)):
            n = unq[i]
            unq_inds = unq_inds_vec[i]
            leaf_predictions = self.tree[n].fitted_model.predict(get_sub(unq_inds, A=Anew, is_boolvec=False),
                                                                 get_cost=get_cost,
                                                                 *leafargs,
                                                                 **leafkwargs)
            if i == 0:
                if leaf_predictions.ndim == 1 and len(unq_inds) == len(leaf_predictions):
                    predictions = np.zeros(num_obs)
                else:
                    predictions = np.zeros((num_obs, leaf_predictions.shape[1]))
            predictions[unq_inds] = leaf_predictions
            # for i,unq_ind in enumerate(unq_inds):
            # predictions[unq_ind] = leaf_predictions[i]

        if return_loc:
            return (predictions, (unq, unq_inds_vec))
        else:
            return (predictions)

    def eval_model_choice(self, Xnew, Anew, Ynew, alpha=None, *leafargs, **leafkwargs):
        '''
        Evaluates the performance of the model on the three dimensions
        '''
        # If Xnew is a pandas data frame, convert it to a numpy array
        if (isinstance(Xnew, pd.core.frame.DataFrame)):
            Xnew = Xnew.values

        if alpha is None:
            alpha = self.alpha_best

        num_obs = Xnew.shape[0]

        unq, unq_inds_vec = self._find_leaf_nodes(0, Xnew, np.array(range(0, num_obs)), alpha);

        # Now that we have found the leaf nodes corresponding to each X observation, use the models in these
        performance = np.zeros((num_obs, 4))
        for i in range(0, len(unq)):
            n = unq[i]
            unq_inds = unq_inds_vec[i]
            performance_dict = self.tree[n].fitted_model.eval_model(
                get_sub(unq_inds, A=Anew, is_boolvec=False),
                get_sub(unq_inds, Y=Ynew, is_boolvec=False))
            performance[unq_inds, 0] = performance_dict["loss"]
            performance[unq_inds, 1] = performance_dict["accuracy"]
            performance[unq_inds, 2] = performance_dict["average_rank"]
            performance[unq_inds, 3] = performance_dict["average_perc"]
            # for i,unq_ind in enumerate(unq_inds):
            # predictions[unq_ind] = leaf_predictions[i]
        return (np.average(performance, axis=0))

    def _error(self, Xnew, Anew, Ynew,
               alpha=None, return_locNerrors=False, init_locNerrors=None, use_pruning_error=True):
        """
        Given data Xnew,Anew outputs errors for each observation (e.g., prediction error) as a function of response data Ynew
        Required: call tree fit() method first
        Uses pruned tree if pruning method has been called, else uses unpruned tree
        Argument alpha controls level of pruning. If not specified, uses alpha trained from the prune() method

        As a step in finding the estimated probabilities for data (Xnew,Anew), this function first finds
        the leaf node locations corresponding to each row of Xnew. It does so by a top-down search
        starting at the root node 0.
        Init_loc is a numpy vector of initial node locations with dimension equal to the number of rows of Xnew.
        If init_loc is specified, algo will search for the leaf node of observation i through a top-down search
        starting at node init_loc[i]. Otherwise, it will start at root node 0.
        - UPDATE: We now encode init_loc NOT using a vector of integers, but rather using (unq,unq_inds_vec), where:
          - unq: a vector giving the unique elemenets of array init_loc
          - unq_inds_vec: unq_inds_vec[i] are the elements of init_loc equal to unq[i], i.e. init_loc[unq_inds_vec[i]] == unq[i]
        If return_loc=True, predict() will also return the leaf node locations for the data, in addition to the prob estimates.
        use_pruning_error (default True) uses the pruning error function in leaf_model.py, rather than the fitting error function
        """

        # If Xnew is a pandas data frame, convert it to a numpy array
        if (isinstance(Xnew, pd.core.frame.DataFrame)):
            Xnew = Xnew.values

        if alpha is None:
            alpha = self.alpha_best

        num_obs = Xnew.shape[0]

        if init_locNerrors is None:
            unq, unq_inds_vec = self._find_leaf_nodes(0, Xnew, np.array(range(0, num_obs)), alpha)
            leaf_errors_lookup = {}
        else:
            init_unq = init_locNerrors[0]
            init_unq_inds_vec = init_locNerrors[1]
            leaf_errors_lookup = init_locNerrors[2]
            unq = []  # unq: a vector giving the unique leaf node indices corresponding to X
            unq_inds_vec = []  # unq_inds_vec: unq_inds_vec[i] are X's observation indices in leaf unq[i], i.e. X[unq_inds_vec[i],:] are in leaf unq[i]
            for i in range(0, len(init_unq)):
                n = init_unq[i]
                unq_inds = init_unq_inds_vec[i]
                # find leaf nodes for all data in node n of the tree
                unq_i, unq_inds_vec_i = self._find_leaf_nodes(n, Xnew[unq_inds, :], unq_inds, alpha);
                # append the data to unq,unq_inds_vec
                unq.extend(unq_i)
                unq_inds_vec.extend(unq_inds_vec_i)

        # Now that we have found the leaf nodes corresponding to each X observation, use the models in these
        # leaf nodes to find the errors
        errors = np.zeros(num_obs)
        new_leaf_errors_lookup = {}
        for i in range(0, len(unq)):
            n = unq[i]
            unq_inds = unq_inds_vec[i]
            if n in leaf_errors_lookup:
                leaf_errors = leaf_errors_lookup[n]
            else:
                Asub, Ysub = get_sub(unq_inds, A=Anew, Y=Ynew, is_boolvec=False)
                if use_pruning_error == True:
                    leaf_errors = self.tree[n].fitted_model.error_pruning(Asub, Ysub)
                else:
                    leaf_errors = self.tree[n].fitted_model.error(Asub, Ysub)
            errors[unq_inds] = leaf_errors
            new_leaf_errors_lookup[n] = leaf_errors

        if (return_locNerrors == True):
            return (errors, (unq, unq_inds_vec, new_leaf_errors_lookup))
        else:
            return (errors)

    def _find_leaf_nodes(self, n, X, data_inds, alpha):
        '''
        Helper function for predict()
        Recursive function for finding the leaf nodes for each observation in X
        Returns unq,unq_inds_vec:
          - unq: a vector giving the unique leaf node indices corresponding to X
          - unq_inds_vec: unq_inds_vec[i] are X's observation indices in leaf unq[i], i.e. X[unq_inds_vec[i],:] are in leaf unq[i]
        Arguments:
          - n: the leaf node containing the data X
          - X: a numpy array representing the covariates
          - data_inds: data_inds[i] gives the index of observation X[i,] in the original dataset passed to the fit() function
          - alpha: pruning parameter
        '''
        # Check if the node is leaf or is already pruned
        # if alpha >= self.tree[n].alpha_thresh:
        #     return [n], [data_inds]
        if self.tree[n].is_leaf:
            return [n], [data_inds]

        split_var = X[:, self.tree[n].split_var_ind]
        child_inds = self.tree[n].child_ind
        if self.tree_params.binary_splits:
            # find split directions (True maps to child_inds[0], False maps to child_inds[1])
            if self.tree[n].is_split_var_numeric:
                dirs = split_var < self.tree[n].split_val
            else:
                dirs = np.in1d(split_var, self.tree[n].split_val)

            if len(np.unique(dirs)) <= 1:
                # unq: a vector giving the unique leaf node indices corresponding to X
                # unq_inds_vec: unq_inds_vec[i] are X's observation indices in leaf unq[i], i.e. X[unq_inds_vec[i],:] are in leaf unq[i]
                unq, unq_inds_vec = self._find_leaf_nodes(child_inds[int(not (dirs[0]))], X, data_inds, alpha)
                return unq, unq_inds_vec
            else:
                # Left split (child_inds[0])
                unq_l, unq_inds_vec_l = self._find_leaf_nodes(child_inds[0], X[dirs, :], data_inds[dirs], alpha);
                # Right split (child_inds[1])
                dirs_opp = np.logical_not(dirs)
                unq_r, unq_inds_vec_r = self._find_leaf_nodes(child_inds[1], X[dirs_opp, :], data_inds[dirs_opp],
                                                              alpha);

                unq_l.extend(unq_r)
                unq_inds_vec_l.extend(unq_inds_vec_r)
                return unq_l, unq_inds_vec_l

        else:
            val2child = self.tree[n].val2child
            children = which_child_multi(split_var,
                                         val2child)

            if (len(np.unique(children)) <= 1):
                unq, unq_inds_vec = self._find_leaf_nodes(children[0], X, data_inds, alpha);
            else:
                # finds unique values and data indices of immediate children
                unq_ch, unq_ch_inds_vec = get_unique_valsNinds(children)

                # unq: a vector giving the unique leaf node indices corresponding to X
                unq = []
                # unq_inds_vec: unq_inds_vec[i] are X's observation indices in leaf unq[i], i.e. X[unq_inds_vec[i],:] are in leaf unq[i]
                unq_inds_vec = []
                for i, ch in enumerate(unq_ch):
                    data_in_ch = unq_ch_inds_vec[i]
                    # finds unique values and data indices of leaves
                    unq_leaves, unq_inds_vec_leaves = self._find_leaf_nodes(ch, X[data_in_ch, :], data_inds[data_in_ch],
                                                                            alpha);
                    unq.extend(unq_leaves)
                    unq_inds_vec.extend(unq_inds_vec_leaves)

            return unq, unq_inds_vec

    def _initialize_pruning(self, verbose):
        """
        Computes the sequence of alpha's in Breiman's pruning algorithm, i.e.
        computes the sequence of subtrees to choose from using cross-validation
        For more details, see section 10.3 of Breiman's Classification and Regression Trees book
        """
        if verbose: print("Initializing Pruning Alphas");
        index_last = len(self.tree)
        ###########################################################
        # PRUNING SUPPORT
        ###########################################################
        # reversed(range(0,index_last)) = index_last-1, index_last-2,...,0
        for t in reversed(range(0, index_last)):
            self.tree[t].alpha_thresh = float("inf");
            if (self.tree[t].is_leaf):
                self.tree[t].N = 1.0;
                self.tree[t].S = self.tree[t].fitted_model_error
                self.tree[t].G = float("inf");
            else:
                child_t = self.tree[t].child_ind;
                self.tree[t].N = sum([self.tree[ch].N for ch in child_t])
                self.tree[t].S = sum([self.tree[ch].S for ch in child_t])
                self.tree[t].g = (self.tree[t].fitted_model_error - self.tree[t].S) / (self.tree[t].N - 1.0);
                if (np.isnan(self.tree[t].g)):
                    print("Warning: NaN encountered in pruning procedure")
                    self.tree[t].g = 0.0;
                min_child_G = min([self.tree[ch].G for ch in child_t])
                self.tree[t].G = min(self.tree[t].g, min_child_G);

        alpha = 0.0
        alpha_seq = np.array([])
        while (True):
            if (self.tree[0].G > alpha):
                alpha_seq = np.append(alpha_seq, alpha)
                alpha = self.tree[0].G

            if (self.tree[0].N == 1.0):
                alpha_seq = np.append(alpha_seq, float("inf"))

                # create sequence of subtrees to test when pruning by
                # taking the geometric mean between each series of points in alpha_seq
                if (len(alpha_seq) == 2):
                    self.alpha_seq = alpha_seq
                else:
                    self.alpha_seq = np.array([0.0] * (len(alpha_seq) - 1))
                    for i in range(0, len(self.alpha_seq)):
                        self.alpha_seq[i] = np.sqrt(alpha_seq[i] * alpha_seq[i + 1])

                self.alpha_best = 0  # alpha_best = 0 (no pruning) until pruning f'n called
                if (verbose == True): print("Done!");
                # we are done with the pruning set-up, so return
                return

            # Find node t in the current subtree with minimal g value, breaking ties by going left
            t = 0
            while (self.tree[t].G < self.tree[t].g):
                child_t = self.tree[t].child_ind
                for ch in child_t:
                    if (self.tree[t].G == self.tree[ch].G):
                        t = ch
                        break

            self.tree[t].alpha_thresh = alpha
            self.tree[t].N = 1.0
            self.tree[t].S = self.tree[t].fitted_model_error
            self.tree[t].G = float("inf");
            while (t > 0):
                t = self.tree[t].parent_ind;
                child_t = self.tree[t].child_ind;
                self.tree[t].N = sum([self.tree[ch].N for ch in child_t])
                self.tree[t].S = sum([self.tree[ch].S for ch in child_t])
                self.tree[t].g = (self.tree[t].fitted_model_error - self.tree[t].S) / (self.tree[t].N - 1.0);
                min_child_G = min([self.tree[ch].G for ch in child_t])
                self.tree[t].G = min(self.tree[t].g, min_child_G)


class _TreeParams(object):
    """
    Object which stores tree parameters
    """
    pass


class _LeafParams(object):
    """
    Object which stores leaf parameters
    """
    pass


class _Node(object):
    '''
    Node object (can be terminal or non-terminal)
    '''

    def __init__(self):
        '''
        NODE ATTRIBUTES
        (note: attribute=None until defined)

        ind: index of node (root has index 0)
        parent_ind: index of parent node (root has parent_ind = None)
        depth: depth of node in tree
        is_leaf: is this node a leaf node?
        data_inds: indices of training data beloning to this node
        fitted_model: the model fitted on data_inds
        fitted_model_error: the weighted sum of the model errors on data_inds, divided by the sum of all weights in the ENTIRE training data
        alpha_thresh: if pruning coefficient alpha < alpha_thresh, continue down tree, else stop

        ADDITIONAL NON-TERMINAL NODE ATTRIBUTES
        #split_var_ind: the index of the chosen feature to split on
        #Binary splits:
          #is_split_var_numeric: is the split variable numeric or categorical
          #split_val: the split value. The tree splits using the question
          #"is x[split_var_ind] < split_val" for numeric variables
          #"is x[split_var_ind] in split_val" for categorical variables (split_val is a subset of split values)
        #Non-binary splits:
          #val2child: a dictionary. val2child[v] yields the child node corresponding to when the splitting variable takes value v
          #child2val: a dictionary. child2val[ch] yields the feature value which maps to child node ch
        #child_ind: the indices of the child nodes. If binary splits, indices = child_ind[0],child_ind[1]

        PRUNING NODE ATTRIBUTES (see Breiman's algo)
        #N
        #S
        #G
        #g
        '''
        self.ind = None
        self.parent_ind = None
        self.depth = None
        self.is_leaf = None
        self.data_inds = None
        self.fitted_model = None
        self.fitted_model_error = None
        self.alpha_thresh = None

        self.split_var_ind = None
        self.is_split_var_numeric = None
        self.split_val = None

        self.val2child = None
        self.child2val = None

        self.child_ind = None

        self.N = None
        self.S = None
        self.G = None
        self.g = None

    def set_attr(self, ind=None, parent_ind=None, depth=None, is_leaf=None,
                 data_inds=None, fitted_model=None, fitted_model_error=None, alpha_thresh=None,
                 split_var_ind=None, is_split_var_numeric=None, split_val=None, val2child=None, child2val=None,
                 child_ind=None):
        if ind is not None:
            self.ind = ind
        if parent_ind is not None:
            self.parent_ind = parent_ind
        if depth is not None:
            self.depth = depth
        if is_leaf is not None:
            self.is_leaf = is_leaf
        if data_inds is not None:
            self.data_inds = data_inds
        if fitted_model is not None:
            self.fitted_model = fitted_model
        if fitted_model_error is not None:
            self.fitted_model_error = fitted_model_error
        if alpha_thresh is not None:
            self.alpha_thresh = alpha_thresh
        if split_var_ind is not None:
            self.split_var_ind = split_var_ind
        if is_split_var_numeric is not None:
            self.is_split_var_numeric = is_split_var_numeric
        if split_val is not None:
            self.split_val = split_val
        if val2child is not None:
            self.val2child = val2child
        if child2val is not None:
            self.child2val = child2val
        if child_ind is not None:
            self.child_ind = child_ind


def get_data_inds(X, data_inds, split_var_ind, split_val, feats_continuous):
    # create child nodes, first performing the chosen split
    Xj_ind = np.asarray(X[data_inds, split_var_ind])
    if feats_continuous[split_var_ind]:
        tmp = (Xj_ind < split_val)
    else:
        tmp = np.in1d(Xj_ind, split_val)
    not_tmp = np.logical_not(tmp)
    l_split_data_inds = np.array(data_inds)[tmp].tolist()
    r_split_data_inds = np.array(data_inds)[not_tmp].tolist()
    return l_split_data_inds, r_split_data_inds, tmp, not_tmp


def _find_best_split_multiway(node, tree_params, leaf_params, verbose, X, A, Y, weights, D, sum_weights,
                              feats_continuous, Xnames, shuffled_root_inds):
    """
    Finds the best multiway split of decision tree node
    """
    if verbose:
        print("###############Splitting on node: " + str(node.ind) + ", Depth: " + str(node.depth) + ", Num obs: " + str(
        len(node.data_inds)))

    # Check if tree is a leaf node, i.e. when
    # (1) max depth has been reached, or
    # (2) minimum weights per node has been reached, or
    # (3) node consists of only data points from one class
    if ((node.depth == tree_params.max_depth)
            or (sum(np.asarray(weights[node.data_inds])) <= tree_params.min_weights_per_node)
            or not are_Ys_diverse(get_sub(node.data_inds, A=None, Y=Y, is_boolvec=False))):
        node.is_leaf = True
        del node.data_inds  # we don't need this anymore, so delete for memory management
        return [node, None]

    # initialize lists which store split data for each feature
    num_features = X.shape[1]

    # split_var_ind yields the index (0-(num_features-1)) of the best feature to split on (scalar)
    split_var_ind = None
    # split_avg_errors yields the avg error when splitting on best feature (scalar)
    split_avg_errors = float("inf")
    # split_ch_avg_errors[ch] yields the avg error in child ch corresponding to splitting on best feature
    split_ch_avg_errors = None
    # split_ch_fitted_model[ch] yields the avg error in child ch corresponding to splitting on best feature
    split_ch_fitted_model = None
    # split_val2child yields the val2child dictionary corresponding to splitting on best feature
    split_val2child = None

    # create debias set if appropriate
    debias_set_size = np.floor(sum(np.asarray(weights[node.data_inds])) * tree_params.frac_debias_set).astype(int)
    # only debias splits if debias_set_size meets the minimum size requirement
    debias_splits = tree_params.debias_splits and (debias_set_size >= tree_params.min_debias_set_size)
    if (debias_splits == False):
        weights_train = np.asarray(weights[:])
        data_inds_train = node.data_inds
    else:
        '''
        Divide data (represented by weight vector "weights") into training set
        and debias set. Equivalent: create two vectors weights_train and weights_debias
        such that weights[node.data_inds] = weights_train[node.data_inds]+weights_debias[node.data_inds]
        '''
        if (verbose == True): print("Node " + str(node.ind) + ": Generating Debias Set")
        # extract elements of shuffled_root_inds that are equal to data_inds, maintaining random
        # order of indices
        tmp = np.in1d(shuffled_root_inds, node.data_inds)
        shuffled_node_inds = shuffled_root_inds[tmp]
        # take first debias_set_size entries of shuffled_node_inds to make debias set
        shuffled_node_inds_debias = shuffled_node_inds[:debias_set_size]
        # use the rest of the data for training
        shuffled_node_inds_train = shuffled_node_inds[debias_set_size:]
        # create weights_debias and weights_train sets
        weights_debias = np.zeros(len(weights))
        weights_train = np.zeros(len(weights))
        counts_debias = shuffled_node_inds_debias.value_counts()
        counts_train = shuffled_node_inds_train.value_counts()
        weights_debias[counts_debias.index.tolist()] = counts_debias
        weights_train[counts_train.index.tolist()] = counts_train
        data_inds_debias = np.array(node.data_inds)[weights_debias[node.data_inds] != 0].tolist()
        data_inds_train = np.array(node.data_inds)[weights_train[node.data_inds] != 0].tolist()

    # for each categorical covariate j, calculate the best possible error splitting on j
    for j in range(0, num_features):
        if verbose:
            print("Node " + str(node.ind) + ": Splitting on covariate " + Xnames[j])
        Xj = np.asarray(X[data_inds_train, j])
        # use all unique values of the covariate observed in the training set as the candidate split points
        unq, unq_inds_vec = get_unique_valsNinds(Xj)
        child2val = unq
        # if there only exists one categorical feature, we cannot split on this covariate
        if (len(child2val) <= 1):
            continue

        # fill in is_ch_legal, ch_avg_errors, ch_fitted_model, val2child
        is_ch_legal = [None] * len(
            child2val)  # if we perform this split j, is_ch_legal[ch] yields whether child ch is legal
        ch_avg_errors = [0.0] * len(child2val)
        ch_fitted_model = [None] * len(child2val)
        val2child = {}
        for ch in range(0, len(child2val)):
            v = child2val[ch]
            val2child[v] = ch

            ch_split_data_inds = np.array(data_inds_train)[unq_inds_vec[ch]].tolist();
            A_ch, Y_ch = get_sub(ch_split_data_inds, A=A, Y=Y, is_boolvec=False)
            weights_ch = np.asarray(weights[ch_split_data_inds]);
            weights_train_ch = weights_train[ch_split_data_inds];

            # check min obs per node condition
            if sum(weights_ch) <= tree_params.min_weights_per_node:
                is_ch_legal[ch] = False
            else:
                is_ch_legal[ch] = True

            leaf_mod_ch = LeafModel(*leaf_params.leafargs, **leaf_params.leafkwargs)

            # .fit() returns 0 or 1 corresponding to whether an error occurred during fitting process
            err_flag = leaf_mod_ch.fit(A_ch, Y_ch, weights_train_ch, fit_init=node.fitted_model,
                                       *leaf_params.leafargs_fit, **leaf_params.leafkwargs_fit)

            if (err_flag == 1):
                is_ch_legal[ch] = False
                ch_avg_errors[ch] = float("inf")  # do not consider this split
            else:
                ch_fitted_model[ch] = leaf_mod_ch
                if (debias_splits == False):
                    # find training set error
                    ch_avg_errors[ch] = np.dot(leaf_mod_ch.error(A_ch, Y_ch), weights_train_ch) / sum_weights

        if (debias_splits == True):
            Xj = np.asarray(X[data_inds_debias, j]);
            unq, unq_inds_vec = get_unique_valsNinds(Xj)
            for i, v in enumerate(unq):
                ch = val2child[v]
                ch_split_data_inds = np.array(data_inds_debias)[unq_inds_vec[i]].tolist();
                A_ch, Y_ch = get_sub(ch_split_data_inds, A=A, Y=Y, is_boolvec=False)
                weights_debias_ch = weights_debias[ch_split_data_inds];
                # below condition means that error occurred in training procedure, hence we should not consider the split
                if ch_fitted_model[ch] is None:
                    ch_avg_errors[ch] = float("inf")
                elif sum(weights_debias_ch) == 0:
                    ch_avg_errors[ch] = 0
                else:
                    ch_avg_errors[ch] = np.dot(ch_fitted_model[ch].error(A_ch, Y_ch), weights_debias_ch) / sum_weights

        '''
        To assess the legality of the split, make sure that:
          (1) all of the child nodes are legal
          (2) when debias_splits = True, the training set contains the same features as the entire set
        '''
        is_split_legal = all(is_ch_legal) and set(np.unique(np.asarray(X[data_inds_train, j]))) == set(
            np.unique(np.asarray(X[node.data_inds, j])))
        if (sum(ch_avg_errors) < split_avg_errors) and is_split_legal:
            split_var_ind = j
            split_avg_errors = sum(ch_avg_errors)
            split_ch_avg_errors = ch_avg_errors
            split_ch_fitted_model = ch_fitted_model
            split_val2child = val2child

    # make sure that at least one eligible split was found
    if split_var_ind is None:
        node.is_leaf = True
        del node.data_inds  # we don't need this anymore, so delete for memory management
        return [node, None]

    node.split_var_ind = split_var_ind
    node.val2child = split_val2child

    # create child nodes, first perfoming the chosen split
    Xj = np.asarray(X[node.data_inds, node.split_var_ind])
    # use all unique values of the covariate observed in the training set as the candidate split points
    unq, unq_inds_vec = get_unique_valsNinds(Xj)

    child_nodes = [None] * len(unq)
    sum_child_errors = 0.0
    # if debias = True, retrain on debias + training set, rewrite error and fitted model
    for i, v in enumerate(unq):
        ch = split_val2child[v]
        child_nodes[ch] = _Node()
        ch_split_data_inds = np.array(node.data_inds)[unq_inds_vec[i]].tolist()
        if debias_splits:
            A_ch, Y_ch = get_sub(ch_split_data_inds,
                                 A=A, Y=Y,
                                 is_boolvec=False)
            weights_ch = np.asarray(weights[ch_split_data_inds]);

            split_ch_fitted_model[ch] = LeafModel(*leaf_params.leafargs, **leaf_params.leafkwargs)

            # .fit() returns 0 or 1 corresponding to whether an error occurred during fitting process
            err_flag = split_ch_fitted_model[ch].fit(A_ch, Y_ch, weights_ch, fit_init=node.fitted_model,
                                                     *leaf_params.leafargs_fit, **leaf_params.leafkwargs_fit)

            if (err_flag == 1):
                split_ch_avg_errors[ch] = float("inf")  # do not consider this split
            else:
                split_ch_avg_errors[ch] = np.dot(split_ch_fitted_model[ch].error(A_ch, Y_ch), weights_ch) / sum_weights

            if (split_ch_avg_errors[ch] == float("inf")):
                node.is_leaf = True
                del node.data_inds  # we don't need this anymore, so delete for memory management
                return [node, None]

        child_nodes[ch].set_attr(data_inds=ch_split_data_inds,
                                 fitted_model=split_ch_fitted_model[ch],
                                 fitted_model_error=split_ch_avg_errors[ch])
        sum_child_errors = sum_child_errors + child_nodes[ch].fitted_model_error

    # if we have exceeded the minimum depth, AND
    # performing such a split would lead to a HIGHER overall error than not splitting (w.r.t. min_diff),
    # then make this a leaf node
    if (node.depth >= tree_params.min_depth):
        if (sum_child_errors - (-1.0 * node.fitted_model_error) >= -1.0 * tree_params.min_diff):
            node.is_leaf = True
            del node.data_inds  # we don't need this anymore, so delete for memory management
            return [node, None]

    node.is_leaf = False
    del node.data_inds  # we don't need this anymore, so delete for memory management
    return [node, child_nodes]
