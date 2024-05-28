
class SaaMethod:
    def __init__(self, solver, **kwargs):
        self.solver = solver
        self.fitted_Y = None
        self.decision = None
        self.optimised = False
    def fit(self, X, Y, **kwargs):
        self.fitted_Y = Y

    def est_decision(self, X, **kwargs):
        """
        Predicts the cost of the input X
        """
        if 'time_limit' in kwargs.keys():
            time_limit = kwargs['time_limit']
        if self.optimised:
            return self.decision
        else:
            self.decision = self.solver.solve_multiple_models(self.fitted_Y, PDO_flag=True)['weights']
            self.optimised = True
            return self.decision
