import pandas as pd
import numpy as np


def sampling_from_sms_data(x, y, sorting_array, pcb_cat_cols,
                           n_ins,
                           instance_size,
                           po_mode=True,
                           seed=114514):
    # checking the size of sorting_array, should be the same as x.shape[0] and y.shape[0], also y should be n*1
    if (sorting_array.shape[0] != x.shape[0]
            or sorting_array.shape[0] != y.shape[0]
            or (len(y.shape) == 2 and y.shape[1] != 1)):
        raise ValueError('The size of sorting_array, x, and y are not consistent.')

    # two different types of input:
    #   -po_x, po_y: just n_ins*instance_size, samples, y is an array
    #   -p_x, p_y: n_ins*instance_size samples, y is a matrix [n_ins, instance_size],
    #   x is a matrix [n_ins * instance_size, n_features], also x is sorted by sorting_array
    for i in range(n_ins):
        np.random.seed(seed + i)
        # get the index of the sorting_array, with replace so this is an amplifying sampling
        idx = np.random.choice(sorting_array.shape[0], instance_size)
        # po_x, po_y
        if i == 0:
            po_x = x[idx]
            po_y = y[idx]
        else:
            po_x = np.vstack((po_x, x[idx]))
            po_y = np.hstack((po_y, y[idx]))
        # p_x, p_y

        # sort the idx by sorting_array
        sort_sequence = np.argsort(sorting_array[idx])
        sorted_idx = idx[sort_sequence]
        px_piece = x[sorted_idx].reshape([1, -1])
        py_piece = y[sorted_idx].reshape([1, instance_size])
        if i == 0:
            p_x = px_piece
            p_y = py_piece
        else:
            p_x = np.vstack((p_x, px_piece))
            p_y = np.vstack((p_y, py_piece))
        # also return the sorted array to archive po_x to p_x
        if i == 0:
            sorting_id_data = sort_sequence
        else:
            sorting_id_data = np.hstack((sorting_id_data, sort_sequence))
    # make a bool array feat_countinous to indicate whether the feature is continuous or categorical in p_x
    feat_countinous = np.array([True] * p_x.shape[1])
    for i in range(instance_size):
        for col in pcb_cat_cols:
            feat_countinous[i * int(x.shape[1] / instance_size) + col] = False
    if po_mode:
        return po_x, po_y, p_x, p_y, sorting_id_data, feat_countinous
    else:
        return p_x, p_y, p_x, p_y, sorting_id_data, feat_countinous




def read_sms_data(file_path,
                  sheet_name="Sheet1"):
    # Read from xlsx file
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # remove 'board id' and make other categorical data to num
    # df = df.drop(['Board ID'], axis=1)
    # cat_col_names = ['SMT machine', 'Product type', 'Process']
    # feature selection:
    df = df.drop(['Board ID','SMT machine', 'Product type'], axis=1)
    cat_col_names = ['Process']
    # update the col_idx_list
    col_idx_list = []
    # make categorical data to num
    for col in cat_col_names:
        # which col it is
        col_idx = df.columns.get_loc(col)
        # make it num
        unique_vals = df[col].unique()
        for i, val in enumerate(unique_vals):
            df.loc[df[col] == val, col] = i
        col_idx_list.append(col_idx)

    # get the total processing time of this order:
    df['total time'] = df['Averaged CT'] * df['Quantity'] * df['Joint Board'] /60/60
    # for PO methods: y is an array of total time
    po_y = df['total time'].values
    po_x = df.drop(['total time', 'Averaged CT'], axis=1).values

    # for SPO and prescriptive methods: y is a matrix of total time, sorted by quantity * program CT
    df['simulated time'] = df['Quantity'] * df['Program CT'] * df['Joint Board']/60/60
    # return po_x, po_y, and df['simulated time']
    return po_x, po_y, df['simulated time'].values, col_idx_list

