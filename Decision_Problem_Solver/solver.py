import numpy as np


class BaseSolver:
    """

    """
    def __init__(self):
        self.dim = 1

    def get_num_decisions(self):
        pass

    def solve_model(self):
        pass

    def solve_multiple_models(self):
        pass

    def get_obj(self):
        pass

    def get_pdo_obj(self, cost_mat, decision, weight,
                    obj_func=None):
        """
        get the PDO objective function given cost matrix
        :param cost_mat:
        :param decision:
        :param weight:
        :return:

        Args:
            obj_func:
        """
        c = 0
        for w, cost_vec in zip(weight, cost_mat):
            c += w * self.get_obj(cost_vec, decision,
                                  obj_func=obj_func)
        return c

    def reshape_y(self, y:np.ndarray):
        # Check the shape of input
        if len(y.shape) == 2:
            if y.shape[1] == self.get_num_decisions():
                return y
            else:
                return y
                #raise ValueError('input y has wrong shape to the dims')
        else:
            # get the number of OR instances in data y, the extra data will be removed
            num_obs = y.shape[0]
            num_ins = int(num_obs/self.or_ins_num)

            # todo notation and the try-exception module
            Y_reshape = np.zeros([num_ins, self.T])
            for i in range(num_ins):
                Y_reshape[i, :] = y[i*self.get_num_decisions(): (i+1)*self.get_num_decisions()]

            return Y_reshape

    def get_or_c(self,
                 y: np.ndarray,
                 i: int):
        """
        Get the ith uncertainty data from the array and reshape it.
        """

        if len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1):
            return y[i * self.get_num_decisions(): (i + 1) * self.get_num_decisions()].reshape([1, -1])
        else:
            return y[i, :]