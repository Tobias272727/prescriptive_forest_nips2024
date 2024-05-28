#!/usr/bin/env python
# coding: utf-8
import time
from Decision_Problem_Solver.solver import BaseSolver
from gurobipy import *
# import pandas as pd
import numpy as np
from enum import Enum
from collections import OrderedDict
from scipy.stats import truncnorm, uniform, norm

"""
First defined the classes of instances:
Job:
"""


class Job:
    def __init__(self, key):
        """
        key: the id of the job
        """
        self.key = key

        # time lists
        self.start_times = []
        self.progress_times = []
        self.due_date = None
        self.release_time = None

        # precedent job's instance(or maybe ids)
        self.precedent_jobs = []
        self.post_jobs = []


class Solver(BaseSolver):
    """
    The solver class of parallel machine scheduling problem.
    """

    def __init__(self,
                 param_dict,
                 reset=False, presolve=False, relax=False,
                 verbose=False,
                 warmstart=False,
                 alg_type=0,
                 obj_func='CompletionTime',
                 time_limit=300, gap=0.0001,
                 n_core=-1,
                 method=-1):
        super().__init__()

        # param_dict is for generating the OR instances
        self.param_dict = param_dict

        # originate the env
        self.position_dict = {}  # position dict
        self.memo = {}
        # build PM env:
        self.obj_func = None
        self.mode_release = False
        self.positions = []
        self.time_limit = time_limit
        self.gap_pct = gap
        self.n_core = n_core
        self.obj_func = obj_func  # currently no OBJ func settings
        self._build_env()

        # options for the solver
        self.relax = relax
        self.verbose = verbose
        self.method = method

        self.presolve = presolve  # using the presolve algorithm to accelerate the process.
        self.reset = reset
        self.warmstart = warmstart
        self.alg_type = alg_type

    def _make_model(self):
        """
        Make the model for the parallel machine scheduling problem
        """
        self.model = Model()
        self.model.Params.OutputFlag = 0
        # set the available cores
        if self.n_core > 0:
            self.model.Params.Threads = self.n_core

    def _build_env(self):
        """
        Build the environment:
        input are defined in self.param_dict:
        - self.stages:
        """

        if self.param_dict is None:
            raise ValueError('need param_dict to build scheduling environment')

        # Job data
        self.n_job = self.param_dict['n_job']
        self.mode_release = self.param_dict['mode_release']
        self.obj_func = self.param_dict['obj_func']
        jobs = []

        for j in range(self.n_job):
            # job object
            job = Job(key=j)
            # job object with id = j
            job_data = self.param_dict['job_data'][j]

            # Feasible machines and precedent_jobs
            job.post_jobs = job_data['post_jobs']
            job.precedent_jobs = job_data['prec_jobs']

            # release time and due date
            job.release_time = job_data['release_time']
            job.due_date = job_data['due_date']
            jobs.append(job)

        # Jobs data
        self.jobs = OrderedDict(
            (i, jobs[i]) for i in range(len(jobs))
        )
        self.memo = {}
        self.max_p = len(jobs)
        self.positions = [i for i in range(self.max_p)]
        ######################################################################
        # 5. position_dict
        self.position_dict = {}
        _position_list = [(j) for j in self.jobs]
        for index, item in enumerate(_position_list):
            self.position_dict[item] = index

    def _create_variables(self, scen_num):
        """
        sample_num: the number of scenarios
        """
        # 0. X_imkp: decision binary var for i th job's kth stage in m machine's pth pos
        self.X_VAR = self.model.addVars(
            (
                (j, p)
                for j in self.jobs
                for p in self.positions
            ),
            name="X",
            vtype=GRB.BINARY,
        )
        # 1. S_p: Start time in pos p
        self.S_VAR = self.model.addVars(
            (
                (s, p)
                for s in range(scen_num)
                for p in self.positions
            ),
            name="S",
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
        )

        # 2. Cjk: job j's completion time in scenario s
        self.C_VAR = self.model.addVars(
            (
                (j, s)
                for j in self.jobs
                for s in range(scen_num)
            ),
            name="C",
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
        )

        # 3. T_VAR: tardiness of job j in scenario s
        self.T_VAR = self.model.addVars(
            (
                (j, s)
                for j in self.jobs
                for s in range(scen_num)
            ),
            name="T",
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
        )

        # 4. a temp var for duedate - Cj
        self.D_VAR = self.model.addVars(
            (
                (j, s)
                for j in self.jobs
                for s in range(scen_num)
            ),
            name="D",
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
        )

    def _create_constraints(self, scenario_data):
        """
        Create constraints for the single machine scheduling problem:
        1. minimum c_j: single solution can be solved by SPT or wSPT, but SAA still NEEDS to be solved by the model
        2. minimum tardiness: single solution can be solved by EDD or wEDD but SAA still NEEDS to be solved by the model
        3. maximum lateness: minimizing maximum cost algo is optimal, but if including the release time,
        the model still needs to be solved by B&B or other methods.
        4. the num of tardy job
        5. total tardiness: by DP or other methods
        """
        scen_num = scenario_data['sample_num']

        # 1. Job processing once
        self.model.addConstrs(
            (
                quicksum(self.X_VAR[j, p] for p in self.positions) == 1
                for j in self.jobs
            ),
            name="JobProcessingOnce"
        )

        self.model.addConstrs(
            (
                quicksum(self.X_VAR[j, p] for j in self.jobs) == 1
                for p in self.positions
            ),
            name="PosOnlyHaveOneJob"
        )
        # If have release time:
        if self.mode_release:
            # 2. completion time constraints
            self.model.addConstrs(
                (
                    self.S_VAR[s, 0] == quicksum(self.X_VAR[j, 0] * self.jobs[j].release_time for j in self.jobs)
                    for s in range(scen_num)
                ),
                name="StartTime_0"
            )
            self.model.addConstrs(
                (
                    self.S_VAR[s, p] >= quicksum(self.X_VAR[j, p] * self.jobs[j].release_time for j in self.jobs)
                    for p in self.positions[1:]
                    for s in range(scen_num)
                ),
                name="StartTime_allRelease"
            )

            self.model.addConstrs(
                (
                    self.S_VAR[s, p] >= self.S_VAR[s, p-1] +
                    quicksum(self.X_VAR[j, p-1] * scenario_data['processing_time'][s, j] for j in self.jobs)
                    for p in self.positions[1:]
                    for s in range(scen_num)
                ),
                name="Starttime_Adjacent"
            )

            self.model.addConstrs(
                (
                    self.C_VAR[j, s] == quicksum(self.S_VAR[s, p] * self.X_VAR[j, p] for p in self.positions)
                    + scenario_data['processing_time'][s, j]
                    for j in self.jobs
                    for s in range(scen_num)
                ),
                name="CompletionTime_all"
            )
        else:
            # 2. Completion time constraints
            self.model.addConstrs(
                (
                    self.S_VAR[s, 0] == 0
                    for s in range(scen_num)
                ),
                name="CompletionTime_0"
            )

            self.model.addConstrs(
                (
                    self.S_VAR[s, p] == self.S_VAR[s, p-1] +
                    quicksum(self.X_VAR[j, p-1] * scenario_data['processing_time'][s, j] for j in self.jobs)
                    for p in self.positions[1:]
                    for s in range(scen_num)
                ),
                name="CompletionTime_1"
            )

            self.model.addConstrs(
                (
                    self.C_VAR[j, s] == quicksum(self.S_VAR[s, p] * self.X_VAR[j, p] for p in self.positions)
                    + scenario_data['processing_time'][s, j]
                    for j in self.jobs
                    for s in range(scen_num)
                ),
                name="CompletionTime_2"
            )
        # Get the Tj
        if self.obj_func == 'Tardiness':
            # Calculate D_var
            self.model.addConstrs(
                (
                    self.D_VAR[j, s] == self.C_VAR[j, s] - self.jobs[j].due_date
                    for j in self.jobs
                    for s in range(scen_num)
                ),
                name="Tardiness0"
            )

            # Get tj
            self.model.addConstrs(
                (
                    self.T_VAR[j, s] == max_(self.D_VAR[j, s], constant=0)
                    for j in self.jobs
                    for s in range(scen_num)
                ),
                name="Tardiness1"
            )

    def _create_objective(self, scenarios_data):
        if self.obj_func == 'Tardiness':
            obj = quicksum(
                self.T_VAR[j, s]
                for j in self.jobs
                for s in range(scenarios_data['sample_num'])
            )
        else:
            obj = quicksum(
                self.C_VAR[j, s]
                for j in self.jobs
                for s in range(scenarios_data['sample_num'])
                )

        self.model.setObjective(obj, GRB.MINIMIZE)

    def get_num_decisions(self):
        return self.n_job

    def _get_decision(self):
        # get decision from the solution
        x_jp = self.model.getAttr('X', self.X_VAR)
        job_schedule = np.zeros([self.get_num_decisions()])

        # job could be list in a random order
        for p in self.positions:
            for j in self.jobs:
                # use 0.5 instead of 1 to avoid the float error
                if x_jp[(j, p)] > 0.5:
                    job_schedule[(self.position_dict[j])] = p
        return job_schedule

    def _call_algorithms(self, data: dict,
                         verbose=False):
        start = time.time()
        if self.alg_type == 0:
            # build and optimise the model
            self._make_model()
            if verbose:
                self.model.Params.OutputFlag = 1

            self._create_variables(data['sample_num'])
            self._create_objective(data)
            self._create_constraints(data)

            init_sol = self._start(data['processing_time'])
            # check if init_sol is feasible
            if np.unique(init_sol).shape[0] != self.n_job:
                init_sol_feas = False
            else:
                init_sol_feas = True

            # todo update precedent mode
            if data['sample_num'] > 1 or not init_sol_feas or (self.mode_release or self.obj_func == 'Tardiness'):
                # Terminal criteria: solving the problem to reach the acceptable gap pct or stop within the time period
                if self.time_limit is not None:
                    self.model.setParam('TimeLimit', self.time_limit)
                if self.gap_pct is not None:
                    self.model.setParam('MIPGap', self.gap_pct)
                else:
                    self.gap_pct = 1

                run_time = 1
                cur_gap = 1
                # A loop to get at least a feasible solution by 10* running the model under time limit.
                while (run_time < 10 and self.model.SolCount == 0):
                    self.model.optimize()
                    if self.model.SolCount > 0:
                        cur_gap = max(0, abs(self.model.ObjVal - self.model.ObjBound) / abs(self.model.ObjVal))
                    if self.time_limit is not None:
                        self.model.setParam('TimeLimit', self.time_limit * run_time)
                    run_time += 1
                # Time limit reached without any feasible solution, tries to enlarge the time to get the solution

                # self.model.setParam('TimeLimit', self.time_limit)
                # if no feasible sols, return 0 vars
                if self.model.SolCount == 0:
                    print("unable to solve the problem within the time_limit times")
                    return_dict = {}
                    return return_dict
                obj_val = self.model.ObjVal
                d = self._get_decision()
            else:
                d = init_sol
                obj_val = self.get_obj(data['processing_time'], d,
                                       verbose=verbose)

            return_dict = {'weights': d,
                           'objective': obj_val}
            return return_dict

        if self.alg_type == 1:

            # use heuristic to solve the algorithm
            return 0

    def _start(self, cost_mat):
        initial_solution \
            = self._heuristic(cost_mat)
        for j in self.jobs:
            p = initial_solution[j]
            self.X_VAR[(j, p)].Start = 1  # Assigning the start value
        return initial_solution

    def _heuristic(self, cost_mat):
        n_jobs = self.n_job
        cost_mat = self.check_c_shape(cost_mat)
        if len(cost_mat.shape) > 1 and cost_mat.shape[0] > 1:
            # multiple samples
            cost_mat = np.mean(cost_mat, axis=0).reshape(1,-1)
        # Initialize the 4D array for decisions
        decision_array = np.zeros([n_jobs])

        # sort the jobs based on the processing time, if saa, use averaged processing time
        if self.obj_func == 'Tardiness' and self.mode_release:
            # use EDD to initialise the solution
            job_order = np.argsort([self.jobs[j].due_date for j in self.jobs])
        elif self.obj_func == 'Tardiness':
            # use dp to get solution ( optimal)
            result, job_order = self.min_tardiness(cost_mat, tuple(range(n_jobs)), 0)
            job_order = np.array(job_order)
        elif self.obj_func == 'CompletionTime':
            job_order = np.argsort(cost_mat)

        for j in self.jobs:
            decision_array[j] = np.where(job_order.reshape(-1) == j)[0][0]

        return decision_array


    def min_tardiness(self,
                      cost_mat,
                      remaining_jobs,
                      current_time):
        if not remaining_jobs:
            return (0, [])
        if (tuple(remaining_jobs), current_time) in self.memo:
            return self.memo[(tuple(remaining_jobs), current_time)]

        min_tardiness = float('inf')

        best_sequence = []
        for j in remaining_jobs:
            next_jobs = list(remaining_jobs)
            next_jobs.remove(j)
            job = self.jobs[j]
            completion_time = current_time + cost_mat[0, j]
            tardiness = max(0, completion_time - cost_mat[0, j])
            result, sequence = self.min_tardiness(cost_mat, tuple(next_jobs), completion_time)
            total_tardiness = tardiness + result
            if total_tardiness < min_tardiness:
                min_tardiness = total_tardiness
                best_sequence = sequence + [j]

        self.memo[(tuple(remaining_jobs), current_time)] = (min_tardiness, best_sequence)
        return min_tardiness, best_sequence




    def solve_model(self,
                    cost_mat,
                    include_solver=True,
                    PDO_flag=False,
                    weight_mat=None,
                    num_instance=None,
                    verbose=False):

        data = {'processing_time': cost_mat}

        if PDO_flag:
            data['sample_num'] = cost_mat.shape[0]
        else:
            data['sample_num'] = 1

        return_dict = self._call_algorithms(data,
                                            verbose=verbose)
        return return_dict

    def get_or_c(self,
                 input_data: np.ndarray,
                 i: int):
        """
        Get the ith uncertainty data from the array and reshape it.
        """
        # Use check_c_shape to check its structure
        checked_data = input_data

        if (len(checked_data.shape) == 1 or
                (len(checked_data.shape) == 2 and checked_data.shape[1] == 1)):
            return checked_data[i * self.get_num_decisions(): (i + 1) * self.get_num_decisions()].reshape([1, -1])
        else:
            return checked_data[i, :]

    def check_c_shape(self, input_data):
        """
        Formulate the processing time data based on the input array's dimensions.
        :param input_data: numpy array representing processing times
        :return: A 4D numpy array of processing times with dimensions [n_samples, n_jobs, n_machines]
        # Example usage
        # input_data = np.array([...]) # Replace with actual data
        # processing_time_matrix = check_c_shape(input_data)
        """

        n_job = len(self.jobs)

        # if dim==1:
        if input_data.ndim == 1:
            if len(input_data) == n_job:
                # Scenario 1: 1D array with n_job size
                return input_data.reshape([1, n_job])
            elif len(input_data) % n_job == 0:
                # Scenario 2: have multiple 1D array with n_job size
                return input_data.reshape(-1, n_job)
            else:
                raise ValueError('Should have n_job shape of input data')

        elif input_data.ndim == 2:
            if input_data.shape[0] == n_job and input_data.shape[1] == 1:
                # 2D array with size (n_job, n_machine)
                return input_data.reshape([1, n_job])
            elif input_data.shape[0] == 1 and input_data.shape[1] == n_job:
                return input_data
            elif input_data.shape[1] == n_job:
                return input_data

        else:
            raise ValueError("Input data dimensions do not match any expected scenario.")

    def solve_multiple_models(self,
                              cost_mat,
                              weight_mat=None,
                              PDO_flag=False,
                              verbose=False):
        """
        Given cost matrix, get the optimal solution of parallel machine scheudling problem
        ::
        cost: 2 dim matrix, if its PDO model, all samples should be inputed and calculated.
        """
        cost_mat = self.check_c_shape(cost_mat)
        # Finish the weight and obj matrix's calculation by the solution algorithm.
        if PDO_flag:
            weights = np.zeros([1, self.get_num_decisions()])
            outcome_dict = self.solve_model(cost_mat,
                                            weight_mat=weights,
                                            PDO_flag=True,
                                            verbose=verbose)
            # get the weights and objective
            weights = outcome_dict['weights']
            objective = outcome_dict['objective']
        else:
            num_samples = cost_mat.shape[0]
            weights = np.empty([num_samples, self.get_num_decisions()], dtype=object)
            objective = np.zeros(num_samples)
            # loop to get each solution
            for i in range(num_samples):
                outcome_dict = self.solve_model(cost_mat[i, :].reshape([1, -1]),
                                                verbose=verbose)  # input cost_i to the parallel machine problem

                weights[i, :] = outcome_dict['weights']
                objective[i] = outcome_dict['objective']
        if verbose: print(objective)
        return {'weights': weights, 'objective': objective}

    def get_obj(self,
                cost_mat: np.ndarray,
                decision: np.ndarray,
                verbose=False):
        """
        Getting the objective value of a give decision with the cost matrix
        """
        # Get data
        cost_mat = cost_mat.reshape([1, self.n_job])
        job_completion_times = np.zeros((self.n_job))

        cur_c = 0
        for p in self.positions:
            # Loop each job to find scheduled job
            for j in self.jobs:
                if decision[self.position_dict[(j)]] == p:
                    if verbose:
                        print('placed job %d in position %d' % (j, p))
                    if self.mode_release:
                        # release time is on
                        if verbose: print('placed job %d in position %d' % (j, p))
                        if verbose: print('start time', max(cur_c, self.jobs[j].release_time))
                        job_completion_times[j] = max(cur_c, self.jobs[j].release_time) + cost_mat[0, j]
                        cur_c = job_completion_times[j]
                    else:
                        job_completion_times[j] = cost_mat[0, j] + cur_c
                        cur_c = job_completion_times[j]
        # The obj
        if self.obj_func == 'Tardiness':
            obj = np.sum([max(0, job_completion_times[j] - self.jobs[j].due_date) for j in self.jobs])
        else:
            obj = np.sum(job_completion_times)
        if verbose: print('obj is:', obj)

        return obj


def gen_data(n_job: int,
             obj_func='CompletionTime',
             release=False,prec_per=0,
             normalise=False,sorted=False,
            seed=0,
             data_dis='normal', deg=3,
             skip_data=False,
             n_sample=1000):
    """
    Generating the SM problem data
    """
    param_dict = {'n_job': int(n_job)}
    # prec job data
    n_prec_job = int(n_job * prec_per)
    if n_prec_job == 0:
        param_dict['mode_prec'] = False
    else:
        param_dict['mode_prec'] = True
    param_dict['obj_func'] = obj_func
    # Release time data
    param_dict['mode_release'] = release

    # job data
    jobs_data = {}
    for i in range(n_job):
        set_seed(seed+i)  # set the seed for each job
        # job data
        job_data = {}
        if i < n_prec_job * 2:
            if i % 2 == 0:
                job_data['post_jobs'] = [i+1]
                job_data['prec_jobs'] = []
            elif i % 2 == 1:
                job_data['prec_jobs'] = [i-1]
                job_data['post_jobs'] = []
        else:
            job_data['prec_jobs'] = []
            job_data['post_jobs'] = []
        # release time and due date
        # release time: from a normal distribution and abs
        job_data['release_time'] = np.abs(np.random.normal(5, 5))
        job_data['due_date'] = np.random.randint(0, 20)

        jobs_data[i] = job_data
    param_dict['job_data'] = jobs_data

    if skip_data:
        return param_dict
    else:
        if data_dis == 'poly':
            X, Y, X_po, Y_po = gen_data_poly(n_sample,
                                             seed=seed,
                                             d=n_job, p=20,
                                             tau=1.0, deg=deg,
                                             verbose=False)
        elif data_dis == 'normal':
            X, Y, X_po, Y_po = gen_data_normal(n_sample, n_job,
                                               seed=seed, sorted=sorted,
                                               normalise=normalise)

        return param_dict, X, Y, X_po, Y_po


def set_seed(seed_value=114514):
    np.random.seed(seed_value)


def gen_data_poly(n_size,
                  seed=0,
                  d=50, p=20,
                  tau=1.0, deg=5,
                  verbose=False):
    """
    # Parameters
    d = 50  # Number of assets
    p = 20  # Number of features, assuming 20 as an exp_runtime
    tau = 1.0  # Noise level parameter, exp_runtime value
    deg = 3  # Degree for the formula, exp_runtime value
    """

    def gen_data_poly_single(d, p,
                             tau, deg,
                             verbose):
        # Step 1: Generate matrix B* (d x p)
        B_star = np.random.binomial(1, 0.5, (d, p))

        # Step 2: Generate factor loading matrix L (50 x 4)
        L = np.random.uniform(-0.0025 * tau, 0.0025 * tau, (d, 4))

        # Step 3: Generate feature vectors x_i
        x_i = np.random.normal(0, 1, (p,))

        # Step 4: Generate incremental return vectors r_i
        r_ij = 0.05 / np.sqrt(p) * (B_star @ x_i) + (0.1) ** (1/ deg)  # deg used as exponent
        r_i = r_ij.reshape(d, 1)  # reshaping to match dimensions

        # Step 5: Generate noise and observed return vectors
        f = np.random.normal(0, 1, (4,))
        noise = np.random.normal(0, 1, (d, 1))
        tilde_r_i = r_i + L @ f.reshape(4, 1) + 0.01 * noise

        # Normalize and transform tilde_r_i to range [0, 10]
        min_r_i = np.min(tilde_r_i)
        max_r_i = np.max(tilde_r_i)
        tilde_r_i = np.round(10 * (tilde_r_i - min_r_i) / (max_r_i - min_r_i), 1)
        tilde_r_i[tilde_r_i <= 0] = 0.1
        if verbose:
            print("Feature vector x_i:", x_i)
            print("Incremental return vector r_i:", r_i)
            print("Observed return vector tilde_r_i:", tilde_r_i)
        return x_i.reshape([1, -1]), tilde_r_i.reshape([1, -1])

    for i in range(n_size):
        set_seed(seed+i)
        x_i, c_i = gen_data_poly_single(d, p, tau, deg, verbose)
        if i == 0:
            X = x_i
            C = c_i
        else:
            X = np.vstack((X, x_i))
            C = np.vstack((C, c_i))
    return X, C, X, C


def gen_data_normal(n_samples, n_jobs,
                  sorted=False,
                  normalise=False, seed=1):

    def sort_jobs(X_row, Y_row):

        sorted_indices = np.argsort(-X_row[:, 1] * X_row[:, 0])  # Negative for descending order
        X_row = X_row[sorted_indices, :]
        Y_row = Y_row[sorted_indices]
        return X_row, Y_row

    # Number of dimensions/features per job
    n_dim = 5
    X = np.zeros((n_samples, n_jobs, n_dim))
    Y = np.zeros((n_samples, n_jobs))
    X_po = np.zeros((n_samples * n_jobs, n_dim))
    Y_po = np.zeros([n_samples * n_jobs])
    # generate data by sample
    for i in range(n_samples):

        for job in range(n_jobs):
            set_seed(seed + i*n_jobs + job)
            # Feature generation using np.random for uniform and normal distributions
            X[i, job, 0] = np.random.uniform(20, 80, 1)  # X_1
            X[i, job, 1] = np.random.uniform(10, 500, 1).astype(int)  # X_2
            X[i, job, 3] = np.random.uniform(10, 40, 1).astype(int)  # X_4
            X[i, job, 4] = np.random.normal(1, 0.5, 1)  # X_5
            noise_term = np.random.uniform(0, 10, 1)  # X_6 (Noise term, not an input)

            # Compute X_3 based on X_1
            X[i, job, 2] = np.random.uniform(0, 1, 1) * X[i, job, 0]  # X_3

            # Compute the mean and standard deviation for Y
            mu = X[i, job, 0]  # mean
            sigma = X[i, job, 2] / (1 + np.exp(X[i, job, 3] - X[i, job, 1]))  # Standard deviation

            # Generate Y from a truncated normal distribution
            a = mu / sigma  # Calculate 'a' for truncnorm
            b = np.inf  # Upper bound for truncnorm
            Y[i, job] = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=1, random_state=seed)
            Y[i, job] = np.round(Y[i, job] * X[i, job, 1] / 60 / 15) / 4


            # Y[i, job] += np.random.normal(0, noise_term)
        # if Y == 0, set it to 0.25
        Y[i, Y[i, :] == 0] = 0.25
        # make a po_Y and po_X
        X[i, :, :], Y[i, :] = sort_jobs(X[i, :, :], Y[i, :])
        X_po[i * n_jobs: (i + 1) * n_jobs, :] = X[i, :, :]
        Y_po[i * n_jobs: (i + 1) * n_jobs] = Y[i, :]


    # normalise and sort data
    # normalise Y: let the min be 0.1 and max be 0.9
    if normalise:
        # each row of Y is normalised
        for i in range(n_samples):
            y_min = np.min(Y[i, :])
            Y[i, :] = Y[i, :] / y_min

    return X.reshape([n_samples, -1]), Y, X_po, Y_po



if __name__ == '__main__':
    param_dict, X, pt_array = gen_data(20, release=False, prec_per=0)
    solver = Solver(param_dict)
    dict_result = solver.solve_multiple_models(pt_array[0:2, :],
                                               PDO_flag=False,
                                               verbose=True)
    print(dict_result)
    # print(solver.get_obj(pt_array[0, :], dict_result['weights'], verbose=True))
    # release is on:
    param_dict2, X, pt_array = gen_data(10, release=True, prec_per=0)
    solver = Solver(param_dict2)
    dict_result = solver.solve_multiple_models(pt_array[0:2, :], verbose=True)
    print(dict_result)
    print(solver.get_obj(pt_array[0, :], dict_result['weights'][0, :], verbose=True))

