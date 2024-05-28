import numpy as np
import hashlib


def _is_split_legal(sub, Xj, min_weights_per_node, weights):
    """
    Checks if the split is legal for a given min_weights_per_node weights
    :param sub: the input indices subset of the data X
    :param Xj: the data X's jth column
    :param min_weights_per_node: the predetermined min_weights_per_node weights
    :param weights: calculated weights as input
    """
    tmp = np.in1d(Xj, sub)
    not_tmp = np.logical_not(tmp)
    weights_l = np.asarray(weights[tmp])
    weights_r = np.asarray(weights[not_tmp])

    # Check min obs per node condition
    if ((sum(weights_l) <= min_weights_per_node)
            or (sum(weights_r) <= min_weights_per_node)):
        return False
    else:
        return True


def which(bool_vec):
    # Given a vector of booleans, returns indices corresponding to which elements are true
    return np.where(bool_vec)[0].tolist()


def fast_avg(x, weights):
    if weights.shape[0] > x.shape[0]:
        weights = weights[:x.shape[0]]  # todo, update weight by OR sample weight
    return (np.dot(x, weights) * 1.0 / sum(weights))


def which_child_multi(split_var, val2child):
    """
    #Carries out multiway splits
    #which child should I go to, given (split_var, val2child)?
    #Here, split_var can be a vector of observations
    """
    children = [None] * len(split_var)
    for i, v in enumerate(split_var):
        if v in val2child:
            children[i] = val2child[v]
        else:
            # we didn't observe this value of split_var in the training data. Send observation down random branch.
            children[i] = val2child.values()[0]
    return children


def get_unique_valsNinds(x):
    """
    # Given an array x, outputs:
    # unq: a vector giving the unique elemenets of array x
    # unq_inds_vec: unq_inds_vec[i] are the elements of x equal to unq[i], i.e. x[unq_inds_vec[i]] == unq[i]
    """
    unq, unq_inv, unq_cnt = np.unique(x, return_inverse=True, return_counts=True)
    unq_inds_vec = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
    return unq, unq_inds_vec


def _is_node_large(node, X):
    """
    check if node is splitting on this node computationally-expensive enough to warrant parallel processing?
    """
    num_obs = len(node.data_inds)
    if num_obs > 20:
        return True
    else:
        return False
#  num_obs = len(node.data_inds)
#  num_features = X.shape[1]
#  #num_split_vals[j] = number of unique values for feature j
#  num_split_vals = [None]*num_features
#  for j in range(0,num_features):
#    Xj = np.asarray(X[node.data_inds,j])
#    num_split_vals[j] = np.floor(len(np.unique(Xj))/2.0).astype("int");
#
#  max_num_split_vals = max(num_split_vals)
#
#  if max_num_split_vals > 10:
#    return True
#  elif (num_obs > 10000) and (max_num_split_vals > 5):
#    return True
#  else:
#    return False

def array_to_hash(arr):
    """
    Converts a NumPy array into a hash representation.
    This method is more efficient for large arrays compared to converting to string.

    :param arr: NumPy array
    :return: Hash representation of the array
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    # Convert the array to bytes and hash it
    arr_bytes = arr.tobytes()
    hash_value = hashlib.sha256(arr_bytes).hexdigest()
    return hash_value

def array_to_str(arr):
    """
    Converts a NumPy array into a string representation.
    This function flattens the array and converts it to a string.
    It's designed to work with arrays of any shape and data type.

    :param arr: NumPy array
    :return: String representation of the array
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    # Flatten the array and convert it to string
    flattened = arr.flatten()
    str_representation = ','.join(map(str, flattened))
    return str_representation


def _path2str(path):
    """
    Converts a path into a string that can be used to specify the OR problem solved and stored in the dict
    """
    if type(path) == list:
        output = []
        for i in range(len(path)):
            if path[i] == 1:
                output.append(i)
        return str(output)
    else:
        if len(path.shape) == 1:
            return str(np.where(path == 1)[0].tolist())
        elif len(path.shape) == 2 and type(path) == np.matrix:
            return str(np.where(path[0, :] == 1)[1].tolist())
        elif len(path.shape) == 2 and type(path) == np.ndarray:
            return str(np.where(path[0, :] == 1)[0].tolist())
        else:
            print ("wrong input dimension")


# get max combination arrays amomg elements of a matrix
def count_elements(matrix):
    counts = {}
    max_count = 0
    max_comb = None
    for row in matrix:
        row_str = _path2str(row)
        if row_str in counts:
            counts[row_str] += 1
        else:
            counts[row_str] = 1
        if counts[row_str] > max_count:
            max_count = counts[row_str]
            max_comb = row
    return max_comb


def max_elements(value_mat, sub=None, weights=None):
    """
    Returns a surrogate decision's id according to the value_mat
    """
    # if weights is not None:
    #     value_mat = value_mat * weights
    # else:
    #     value_mat = value_mat

    # Mask the value_mat rows based on sub and sum along the columns
    if sub is None:
        row_sums = value_mat.sum(axis=1)
    else:
        row_sums = value_mat[sub].sum(axis=1)

    # Find the index of the minimum sum in the filtered rows
    if row_sums.size > 0:
        min_index = np.argmin(row_sums)
        # Retrieve the original index of the minimum sum
        if sub is None:
            opt_s = min_index
        else:
            opt_s = np.where(sub)[0][min_index]
    else:
        opt_s = None

    return opt_s

    # opt_v = np.inf
    # opt_s = None
    # # LOOP TO GET S AND V
    # for i, s in enumerate(sub):
    #     if s:
    #         # use this as decision
    #         v = sum(value_mat[i, :])
    #         if v < opt_v:
    #             opt_v = v
    #             opt_s = i
    # return opt_s
