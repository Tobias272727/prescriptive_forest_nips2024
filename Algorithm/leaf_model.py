"""
Helper class for mtp.py

Defines the leaf nodes of the tree, specifically
- the computation of the predicted cost vectors and decisions within the given leaf of the tree
- the SPO/MSE loss from using the predicted decision within the leaf
"""

import numpy as np
from .tree_utils import count_elements, max_elements, array_to_hash
# from scipy.spatial import distance

'''
mtp.py depends on the classes and functions below. 
These classes/methods are used to define the model object in each leaf node,
as well as helper functions for certain operations in the tree fitting procedure.

Summary of methods and functions to specify:
  Methods as a part of class LeafModel: fit(), predict(), to_string(), error(), error_pruning()
  Other helper functions: get_sub(), are_Ys_diverse()
  
'''


class LeafModel(object):
    """
    LeafModel: the model used in each leaf.
    Has five methods: fit, predict, to_string, error, error_pruning
    SPO_weight_param: number between 0 and 1:
    Error metric: SPO_loss*SPO_weight_param + MSE_loss*(1-SPO_weight_param)
    """
    # Any additional args passed to mtp's init() function are directly passed here
    def __init__(self, *args, **kwargs):
        self.SPO_weight_param = kwargs["SPO_weight_param"]
        self.SPO_full_error = kwargs["SPO_full_error"]
        self.SPO_loss_bound = 1
        self.MSE_loss_bound = 1

        # The OR solver
        try:
            self.get_obj = kwargs["solver"].make_get_obj()
        except AttributeError:
            self.get_obj = kwargs["solver"].get_obj

        # A solver function from the solver class. (Should not include gurobi solver if the mode runs in parallel)
        self.solve_multiple_models = kwargs["solver"].solve_multiple_models
        self.uncertain_num = kwargs["solver"].get_num_decisions()
        self.reshape_y = kwargs["solver"].reshape_y
        self.get_or_c = kwargs["solver"].get_or_c

        # if mode is IPTB, the decision is a prescriptive decision in this node or leaf
        if kwargs['pred_mode'] == 'IPTB':
            self.prescriptive = True
            self.replace_cost = False
            self.saa_optimised = False
        elif kwargs['pred_mode'] == 'SPO-M':
            self.prescriptive = False
            self.replace_cost = False
            self.saa_optimised = False
        else:
            self.replace_cost = True
            self.prescriptive = False
            self.saa_optimised = False
        # data
        self.sub = None
        self.Y = None
        if self.prescriptive:
            self.decision = None
            self.mu = None
            self.D = None

        # obj_dict: a dict obj
        self.obj_dict = {}

    def fit(self, Y, A,
            weights,
            sub=None,
            D=None, mu=None,
            fit_init=None,
            refit=False,
            SPO_loss_bound=None,
            MSE_loss_bound=None):
        """
        This function trains the leaf node model on the data (A,Y,weights).

        A and Y can take any form (lists, matrices, vectors, etc.). For our applications, I recommend making Y
        the response data (e.g., choices) and A alternative-specific data (e.g., features, choice sets)

        weights: a numpy array of case weights. Is 1-dimensional, with weights[i] yielding
          weight of observation/customer i. If you know you will not be using case weights
          in your particular application, you can ignore this input entirely.

        Returns 0 or 1.
          0: No errors occurred when fitting leaf node model
          1: An error occurred when fitting the leaf node model (probably due to insufficient data)
        If fit returns 1, then the tree will not consider the split that led to this leaf node model

        fit_init is a LeafModel object which represents a previously-trained leaf node model.
        If specified, fit_init is used for initialization when training this current LeafModel object.
        Useful for faster computation when fit_init's coefficients are close to the optimal solution of the new data.

        For those interested in defining their own leaf node functions:
          (1) It is not required to use the fit_init argument in your code
          (2) All edge cases must be handled in code below (ex: arguments
              consist of a single entry, weights are all zero, Y has one unique choice, etc.).
              In these cases, either hard-code a model that works with these edge-cases (e.g.,
              if all Ys = 1, predict 1 with probability one), or have the fit function return 1 (error)
          (3) Store the fitted model as an attribute to the self object. You can name the attribute
              anything you want (i.e., it does not have to be self.model_obj and self.model_coef below),
              as long as its consistent with your predict_prob() and to_string() methods

        Any additional args passed to mtp's fit() function are directly passed here
        """
        # no need to refit this model since it is already fit to optimality
        # note: change this behavior if debias=TRUE
        if refit:
            return 0

        # sub: the indice of
        self.sub = sub
        self.Y = Y
        self.A = A
        if self.prescriptive:
            self.mu = mu
            self.D = D
            if self.mu is None or D is None:
                raise ValueError('D or mu must be an input when PDO is enabled.')

        # SPO loss bound
        self.SPO_loss_bound = SPO_loss_bound
        self.MSE_loss_bound = MSE_loss_bound

        def fast_row_avg(X, _weights,
                         sub):
            """
            The average function of data in this leaf.
            return the cost_mat for OR solver. and the mean if prediction should be made seperately
            """
            # todo check this one
            # if the input is separate input
            if sub is not None:
                X_temp = X.copy()
                if len(X.shape) == 1 or X.shape[1] == 1:
                    mean = np.matmul(_weights[sub], X[sub]) / sum(_weights[sub])
                    X_temp[sub] = mean
                    X_temp_reshaped = self.reshape_y(X_temp)
                    weights_temp = np.ones([X_temp_reshaped.shape[0]])
                    return (np.matmul(weights_temp, X_temp_reshaped)/sum(weights_temp)).reshape(-1), mean
                else:
                    mean = None
                    return (np.matmul(_weights[sub], X[sub])/sum(_weights[sub])).reshape(-1), mean
            else:
                mean = None
                return (np.matmul(_weights, X)/sum(_weights)).reshape(-1), mean

        # if PDO mod, the model will get the SAA solution as decision for this leaf
        if self.prescriptive:
            # Get the partial data Y
            self.d_index = max_elements(self.mu)
            self.decision = self.D[self.d_index, :]
            self.mean = np.mean(self.Y)  # no mean value used for this method
        else:
            # if no observations are mapped to this leaf, then assign any feasible cost vector here
            if sum(weights) == 0:
                self.mean_cost, self.mean = fast_row_avg(Y, np.ones(weights.shape), sub)
            # TODO: check this
            else:
                self.mean_cost, self.mean = fast_row_avg(Y, weights, sub)

            # get the decision.
            if self.replace_cost:
                # todo update this for spo-s
                self.decision = None
                self.mean_cost, self.mean = fast_row_avg(Y, weights, sub)
            else:
                self.decision \
                    = self.solve_multiple_models(self.mean_cost.reshape(1, -1))['weights'].reshape(-1)
                self.mean_cost, self.mean = fast_row_avg(Y, weights, sub)



    '''
    This function applies model from fit() to predict choice data given new data A.
    Returns a list/numpy array of choices (one list entry per observation, i.e. l[i] yields prediction for ith obs.).
    Note: make sure to call fit() first before this method.
    
    Any additional args passed to mtp's predict() function are directly passed here
    '''
    def predict(self, A, get_cost=False, *args, **kwargs):
        if get_cost:
            #  Returns predicted cost corresponding to this leaf node
            return np.array([self.mean_cost]*len(A))
        else:
            #  Returns predicted decision corresponding to this leaf node
            if self.prescriptive:
                return np.array([self.sub]*len(A))
            else:
                return np.array([self.decision]*len(A))
            #return np.array([self.decision]*len(A))
    '''
    This function outputs the errors for each observation in pair (A,Y).  
    Used in training when comparing different tree splits.
    Ex: mean-squared-error between observed data Y and predict(A)
    
    How to pass additional arguments to this function: simply pass these arguments to the init()/fit() functions and store them
    in the self object.
    '''

    def error(self, A=None, Y=None):
        def MSEloss(C, Cpred):
            #return distance.cdist(C, Cpred, 'sqeuclidean').reshape(-1)
            MSE = (C**2).sum(axis=1)[:, None] - 2 * C.dot(Cpred.transpose()) + ((Cpred**2).sum(axis=1)[None, :])
            return MSE.reshape(-1)

        def SPOloss(C, decision):

            # use the OR problem sovler's get obj function to get the obj val.
            losses = np.zeros([C.shape[0]])

            for i in range(C.shape[0]):
                # Get the or_c of ith sample
                c = self.get_or_c(C, i)
                  #, **self.or_args)
                # losses[i] = self.get_obj(c, decision)
                d_str = array_to_hash(decision)
                c_str = array_to_hash(c)
                if d_str in self.obj_dict.keys():
                    if c_str in self.obj_dict[d_str].keys():
                        losses[i] = self.obj_dict[d_str][c_str]
                    else:
                        losses[i] = self.get_obj(c, decision)
                        self.obj_dict[d_str][c_str] = losses[i]
                else:
                    self.obj_dict[d_str] = {c_str: self.get_obj(c, decision)}

            return losses

        if self.SPO_weight_param == 0.0:
            MSE_loss = MSEloss(Y, self.mean_cost.reshape(1, -1))
            return MSE_loss
        else:
            if Y is None and A is None:
                # Y is none means this solution is from the historical decision
                obj_vals = self.mu[self.d_index, :]
                # if using SPO_full_error, the error will use the full-informated objective value to calculate value.
                if self.SPO_full_error:
                    SPO_loss = obj_vals
                else:
                    # use the full-informated objective value to normalise the vals
                    # obj_vals-a/obj_vals
                    SPO_loss = (obj_vals - self.A.reshape([-1])) / self.A.reshape([-1])
                    SPO_loss = np.array(SPO_loss).reshape(obj_vals.shape)
                    # temp code for debug: check if there is any negative value in the SPO_loss ( round to 6 digits)
                    if np.any(SPO_loss < -0.0000001):
                        SPO_loss
            # Y not none
            else:
                obj_vals = SPOloss(Y, self.decision)
                # if using SPO_full_error, the error will use the full-informated objective value to calculate value.
                if self.SPO_full_error:
                    SPO_loss = obj_vals
                else:
                    # in this case, input A has identical shape to obj_vals
                    # SPO_loss = obj_vals - A.reshape([-1])
                    SPO_loss = [(obj_vals[idx] - A.reshape([-1])[idx])/A.reshape([-1])[idx] for idx in range(A.shape[0])]
                    SPO_loss = np.array(SPO_loss).reshape(obj_vals.shape)
            return SPO_loss
            # if self.SPO_weight_param == 1.0:
            #     return SPO_loss
            # else:
            #     MSE_loss = MSEloss(Y, self.mean_cost.reshape(1, -1))
            #     return self.SPO_weight_param * SPO_loss / self.SPO_loss_bound + (1.0-self.SPO_weight_param) * MSE_loss / self.MSE_loss_bound

    '''
    This function outputs the errors for each observation in pair (A,Y).  
    Used in pruning to determine the best tree subset.
    Ex: mean-squared-error between observed data Y and predict(A)
    
    How to pass additional arguments to this function: simply pass these arguments to the init()/fit() functions and store them
    in the self object.
    '''
    def error_pruning(self,A, Y):
        return self.error(A, Y)

    '''
    This function returns the string representation of the fitted model
    Used in traverse() method, which traverses the tree and prints out all terminal node models
    
    Any additional args passed to mtp's traverse() function are directly passed here
    '''
    def to_string(self, *leafargs, **leafkwargs):
        return "Mean cost vector: \n" + str(self.mean_cost) +"\n"+"decision: \n"+str(self.decision)


def get_sub(data_inds,
            A=None, Y=None,
            or_ins_num=1,
            is_boolvec=False):
    """
    Given attribute data A, choice data Y, and observation indices data_inds,
    extract those observations of A and Y corresponding to data_inds

    If only attribute data A is given, returns A.
    If only choice data Y is given, returns Y.

    Used to partition the data in the tree-fitting procedure
    """
    # If only A is provided
    if A is not None and Y is None:
        return A[data_inds]

    # If only Y is provided
    if Y is not None and A is None:
        return Y[data_inds]

    # A and Y has value
    if or_ins_num > 1:
        if is_boolvec:
            # Use boolean indexing to filter indices
            or_sample_inds = np.arange(len(data_inds))[data_inds] // or_ins_num
        else:
            or_sample_inds = np.arange(Y.shape[0]) // or_ins_num

        # Utilize numpy's unique function to avoid duplicates efficiently
        or_sample_inds = np.unique(or_sample_inds)
        # or_sample_inds = []
        # for i in range(0, Y.shape[0]):
        #     if not is_boolvec or data_inds[i]:
        #         or_inds = int(i/or_ins_num)
        #         if or_inds in or_sample_inds:
        #             continue
        #         else:
        #             or_sample_inds.append(or_inds)

        return A[or_sample_inds], Y[data_inds]
    else:
        return A[data_inds], Y[data_inds]


'''
This function takes as input choice data Y and outputs a boolean corresponding
to whether all of the choices in Y are the same. 

It is used as a test for whether we should make a node a leaf. If are_Ys_diverse(Y)=False,
then the node will become a leaf. Otherwise, if the node passes the other tests (doesn't exceed
max depth, etc), we will consider splitting on the node.
'''
def are_Ys_diverse(Y):
    #return False iff all cost vectors (rows of Y) are the same
    if len(Y.shape) == 1:
        tmp = [len(np.unique(Y[:]))]
    else:
        tmp = [len(np.unique(Y[:,j])) for j in range(Y.shape[1])]
    return (np.max(tmp) > 1)

