
# This file is part of the
#   SimulAI Project (https://github.com/carosaav/SimulAI).
# Copyright (c) 2020, Perez Colo Ivo, Pirozzo Manuel Bernardo,
# Carolina Saavedra Sueldo
# License: MIT
#   Full Text: https://github.com/carosaav/SimulAI/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================


from abc import ABCMeta, abstractmethod
from .interface import Com
import numpy as np


# ============================================================================
# INPUT-OUTPUT VARIABLES
# ============================================================================


class DiscreteVariable:
    """Initialize the Tecnomatix Plant Simulation Variables that will be used
    in the AI ​​method.
    Up to 4 discrete variables are allowed in the problem which can form up to
    625 possible states in the algorithm.
    For example, if 4 variables are chosen, each of them can take 5 possible
    values and states formed will be S = (Var1, Var2, Var3, Var4).

        Parameters
    ----------
    name: str
        Name of the Variable.
    lower_limit: positive int
        Lower limit of the Variable. Should be a positive integer.
    upper_limit: positive int
        Upper limit of the Variable. Should be a positive integer.
    step: positive int
        Step of the Variable. Should be a positive integer.
    path: str
        Path of the Variable in Tecnomatix Plant Simulation.
    """

    def __init__(self, name, lower_limit, upper_limit, step, path):
        self.name = name
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.step = step
        self.path = path

        if not isinstance(self.name, str):
            raise TypeError("Name: Argument must be a string.")
        if not isinstance(self.lower_limit, int):
            raise TypeError("Lower Limit: Argument must be an integer.")
        if not isinstance(self.upper_limit, int):
            raise TypeError("Upper Limit: Argument must be an integer.")
        if not isinstance(self.step, int):
            raise TypeError("Step: Argument must be an integer.")
        if not isinstance(self.path, str):
            raise TypeError("Path: Argument must be a string.")


class OutcomeVariable:
    """Initialize the Tecnomatix Plant Simulation Variables that will be used
    as the output to optimize in the AI ​​method.
    These variables must be stored in a Data Table.
    The chosen column from which to extract the results and the number of
    rows it has must be indicated.

        Parameters
    ----------
    name: str
        Name of the Variable.
    path: str
        Path of the Variable in Tecnomatix Plant Simulation.
    column: positive int
        Column of the table where the result is stored.
        Should be a positive integer.
    num_rows: positive int
        Number of rows in the results table. Should be a positive integer.
    """

    def __init__(self, name, path, column, num_rows):
        self.name = name
        self.path = path
        self.column = column
        self.num_rows = num_rows

        if not isinstance(self.name, str):
            raise TypeError("Name: Argument must be a string.")
        if not isinstance(self.path, str):
            raise TypeError("Path: Argument must be a string.")
        if not isinstance(self.column, int):
            raise TypeError("Column: Argument must be an integer.")
        if not isinstance(self.num_rows, int):
            raise TypeError("Num_rows: Argument must be an integer.")


# ============================================================================
# PLANTS
# ============================================================================


class Plant(metaclass=ABCMeta):
    def __init__(self, method):
        self.method = method
        method.register(self)

    def connection(self):
        file_name = self.get_file_name_plant()
        self.connect = Com(file_name)
        return self.connect.connection()

    @abstractmethod
    def get_file_name_plant(self):
        pass

    @abstractmethod
    def process_simulation(self):
        pass

    @abstractmethod
    def update(self, data):
        pass


class BasePlant(Plant):
    def __init__(self, method, v_i, v_o, filename, modelname="Model"):
        Plant.__init__(self, method)

        self.v_i = v_i
        self.v_o = v_o
        self.filename = filename
        self.modelname = modelname

    def get_file_name_plant(self):
        return self.filename

    def update(self, data):
        for idx, x in enumerate(self.v_i):
            self.connect.setValue(x.path, data[idx])

        self.connect.startSimulation(".Models.{}".format(self.modelname))

        r = 0
        for idx, x in enumerate(self.v_o):
            a_idx = np.zeros(x.num_rows)
            for h in range(1, x.num_rows + 1):
                a_idx[h - 1] = self.connect.getValue(
                        x.path + str([x.column, h]))
            b_idx = np.sum(a_idx)
            r += b_idx / len(self.v_o)

        self.connect.resetSimulation(".Models.{}".format(self.modelname))
        return r

    def process_simulation(self):
        if self.connection():
            self.connect.setVisible(True)
            self.method.process()


# ============================================================================
# METHODS
# ============================================================================


class AutonomousDecisionSystem(metaclass=ABCMeta):
    def __init__(self):
        self.method = ""

    def register(self, who):
        self.subscriber = who

    @abstractmethod
    def process(self):
        pass


class BaseMethod(AutonomousDecisionSystem):
    """Initialize the states, actions and Q table required to implement
    reinforcement learning algorithms, like Q-learning and SARSA.
    The Q table has a maximum of 625 rows, that is, up to 625 states are
    supported. These states are made up of 1 to 4 variables of the Tecnomatix
    Plant Simulation.
    Actions also depend on the chosen variables and their steps.
    The reward function depends on the results defined in the respective plant
    class.
    """

    def __init__(self, v_i, alfa, gamma, epsilon, episodes_max, steps_max, seed):
        AutonomousDecisionSystem.__init__(self)

        self.v_i = v_i
        self.s = []
        self.a = []

        # reinforcement learning parameters
        self.alfa = alfa
        self.gamma = gamma
        self.epsilon = epsilon

        # number of episodes
        self.episodes_max = episodes_max

        # number of steps
        self.steps_max = steps_max

        # initialize reward per episode
        self.r_episode = np.arange(self.episodes_max, dtype=float)

        if seed is not None:
            np.random.seed(seed)

    # arrays for states and actions
    def arrays(self):
        for idx, x in enumerate(self.v_i):
            self.s_idx = np.arange(
                x.lower_limit, x.upper_limit + x.step, x.step)
            self.a_idx = np.array([-x.step, 0, x.step])
            self.s.append(self.s_idx)
            self.a.append(self.a_idx)

    # initialize states, actions and Q table
    def ini_saq(self):
        self.arrays()
        n = []
        m = []
        for idx, x in enumerate(self.s):
            n.append(x.shape[0])
        for idx, x in enumerate(self.a):
            m.append(x.shape[0])

        if len(self.v_i) == 1:
            self.S = self.s[0]
            self.actions = self.a[0]
        elif len(self.v_i) == 2:
            self.S = np.column_stack((
                np.repeat(self.s[0], n[1]),
                np.tile(self.s[1], n[0])))
            self.actions = np.column_stack((
                np.repeat(self.a[0], m[1]),
                np.tile(self.a[1], m[0])))
        elif len(self.v_i) == 3:
            self.S = np.column_stack((
                np.repeat(self.s[0], n[1] * n[2]),
                np.tile(np.repeat(self.s[1], n[2]), n[0]),
                np.tile(self.s[2], n[0] * n[1])))
            self.actions = np.column_stack((
                np.repeat(self.a[0], m[1] * m[2]),
                np.tile(np.repeat(self.a[1], m[2]), m[0]),
                np.tile(self.a[2], m[0] * m[1])))
        elif len(self.v_i) == 4:
            self.S = np.column_stack((
                np.repeat(self.s[0], n[1] * n[2] * n[3]),
                np.tile(np.repeat(self.s[1], n[2] * n[3]), n[0]),
                np.tile(np.repeat(self.s[2], n[3]), n[1] * n[0]),
                np.tile(self.s[3], n[0] * n[1] * n[2])))
            self.actions = np.column_stack((
                np.repeat(self.a[0], m[1] * m[2] * m[3]),
                np.tile(np.repeat(self.a[1], m[2] * m[3]), m[0]),
                np.tile(np.repeat(self.a[2], m[3]), m[1] * m[0]),
                np.tile(self.a[3], m[0] * m[1] * m[2])))
        else:
            raise Exception("The method admits 4 variables or less")

        if self.S.shape[0] > 625:
            raise Exception("The method supports up to 625 states")

        self.Q = np.zeros((self.S.shape[0], self.actions.shape[0]))

        return self.Q, self.S, self.actions


class Qlearning(BaseMethod):
    def __init__(self, v_i, alfa=0.10, gamma=0.90, epsilon=0.10,
                 episodes_max=100, steps_max=100, seed=None):
        super().__init__(v_i, alfa, gamma, epsilon, episodes_max, steps_max, seed)

    # choose action
    def choose_action(self, row):
        p = np.random.random()
        if p < (1 - self.epsilon):
            i = np.argmax(self.Q[row, :])
        else:
            i = np.random.choice(self.actions.shape[0])
        return i

    # reinforcement learning process
    def process(self):
        self.ini_saq()
        for n in range(self.episodes_max):
            S0 = self.S[0]
            t = 0
            r_acum = 0
            res0 = self.subscriber.update(S0)
            while t < self.steps_max:
                # find k index of current state
                for k in range(self.S.shape[0]):
                    for i in range(len(self.v_i)):
                        if self.S[k][i] == S0[i]:
                            break
                # choose action from row k
                j = self.choose_action(k)
                # update state
                Snew = S0 + self.actions[j]
                # limites
                for idx, x in enumerate(self.v_i):
                    if Snew[idx] > x.upper_limit:
                        Snew[idx] -= x.step
                    elif Snew[idx] < x.lower_limit:
                        Snew[idx] += x.step
                # update simulation result
                res1 = self.subscriber.update(Snew)
                # reward
                if res1 < res0:
                    r = 1
                else:
                    r = 0
                # find index of new state
                for z in range(self.S.shape[0]):
                    for i in range(len(self.v_i)):
                        if self.S[z][i] == Snew[i]:
                            break
                # update Q table
                self.Q[k, j] = self.Q[k, j]
                + self.alfa * (r + self.gamma * np.max(
                            self.Q[z, :]) - self.Q[k, j])
                # update parameters
                t += 1
                S0 = Snew
                res0 = res1
                r_acum = r_acum + r
                self.r_episode[n] = r_acum
        return self.r_episode


class Sarsa(BaseMethod):
    def __init__(self, v_i, alfa=0.10, gamma=0.90, epsilon=0.10,
                 episodes_max=100, steps_max=100, seed=None):
        super().__init__(v_i, alfa, gamma, epsilon, episodes_max, steps_max)

    # choose action
    def choose_action(self, row):
        p = np.random.random()
        if p < (1 - self.epsilon):
            i = np.argmax(self.Q[row, :])
        else:
            i = np.random.choice(self.actions.shape[0])
        return i

    # reinforcement learning process
    def process(self):
        self.ini_saq()
        for n in range(self.episodes_max):
            S0 = self.S[0]
            A0 = self.choose_action(0)
            t = 0
            r_acum = 0
            res0 = self.subscriber.update(S0)
            while t < self.steps_max:
                # find k index of current state
                for k in range(self.S.shape[0]):
                    for i in range(len(self.v_i)):
                        if self.S[k][i] == S0[i]:
                            break
                # update state
                Snew = S0 + self.actions[A0]
                # limites
                for idx, x in enumerate(self.v_i):
                    if Snew[idx] > x.upper_limit:
                        Snew[idx] -= x.step
                    elif Snew[idx] < x.lower_limit:
                        Snew[idx] += x.step
                # update simulation result
                res1 = self.subscriber.update(Snew)
                # reward
                if res1 < res0:
                    r = 1
                else:
                    r = 0
                # find index of new state
                for z in range(self.S.shape[0]):
                    for i in range(len(self.v_i)):
                        if self.S[z][i] == Snew[i]:
                            break
                # choose new action
                Anew = self.choose_action(z)
                # update Q table
                self.Q[k, A0] = self.Q[k, A0]
                + self.alfa * (r +
                               self.gamma * self.Q[z, Anew] - self.Q[k, A0])
                # update parameters
                t += 1
                S0 = Snew
                A0 = Anew
                res0 = res1
                r_acum = r_acum + r
                self.r_episode[n] = r_acum
        return self.r_episode
