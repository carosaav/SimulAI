
# This file is part of the
#   SimulAI Project (https://github.com/carosaav/SimulAI).
# Copyright (c) 2020, Perez Colo Ivo, Pirozzo Bernardo Manuel,
# Carolina Saavedra Sueldo
# License: MIT
#   Full Text: https://github.com/carosaav/SimulAI/blob/master/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""Plant simulation with autonomous decision system."""

# ============================================================================
# IMPORTS
# ============================================================================

from .interface import Com
from abc import ABCMeta, abstractmethod
import numpy as np
import attr


# ============================================================================
# INPUT-OUTPUT VARIABLES
# ============================================================================


@attr.s
class DiscreteVariable:
    """Initialize the input Tecnomatix Plant Simulation Variables.

    These variables will be used in the AI ​​method.
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

    name = attr.ib()
    lower_limit = attr.ib()
    upper_limit = attr.ib()
    step = attr.ib()
    path = attr.ib()

    @name.validator
    def _validate_name(self, attribute, value):
        """Name validator.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        if not isinstance(value, str):
            raise TypeError("Name: Argument must be a string.")

    @lower_limit.validator
    def _validate_lower_limit(self, attribute, value):
        """Lower limit validator.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        if not isinstance(value, int):
            raise TypeError("Lower Limit: Argument must be an integer.")
        if value < 0:
            raise ValueError("Lower limit: Argument must be higher than 0.")

    @upper_limit.validator
    def _validate_upper_limit(self, attribute, value):
        """Upper limit validator.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        if not isinstance(value, int):
            raise TypeError("Upper Limit: Argument must be an integer.")
        if value < 0:
            raise ValueError("Upper Limit: Argument must be higher than 0.")

    @step.validator
    def _validate_step(self, attribute, value):
        """Step validator.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        if not isinstance(value, int):
            raise TypeError("Step: Argument must be an integer.")
        if value < 0:
            raise ValueError("Step: Argument must be higher than 0.")

    @path.validator
    def _validate_path(self, attribute, value):
        """Path validator.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        if not isinstance(value, str):
            raise TypeError("Path: Argument must be a string.")


@attr.s
class OutcomeVariable:
    """Initialize the output Tecnomatix Plant Simulation Variables.

    These variables will be used in the AI ​​method and must be stored in a
    Data Table.
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

    name = attr.ib()
    path = attr.ib()
    column = attr.ib()
    num_rows = attr.ib()

    @name.validator
    def _validate_name(self, attribute, value):
        """Name validator.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        if not isinstance(value, str):
            raise TypeError("Name: Argument must be a string.")

    @path.validator
    def _validate_path(self, attribute, value):
        """Path validator.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        if not isinstance(value, str):
            raise TypeError("Path: Argument must be a string.")

    @column.validator
    def _validate_column(self, attribute, value):
        """Columns validator.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        if not isinstance(value, int):
            raise TypeError("Column: Argument must be an integer.")
        if value < 0:
            raise ValueError("Column: Argument must be higher than 0.")

    @num_rows.validator
    def _validate_num_rows(self, attribute, value):
        """Namber of rows validator.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        if not isinstance(value, int):
            raise TypeError("Num Rows: Argument must be an integer.")
        if value < 0:
            raise ValueError("Num Rows: Argument must be higher than 0.")


# ============================================================================
# PLANTS
# ============================================================================


@attr.s
class Plant(metaclass=ABCMeta):
    """Metaclass to generate various simulated manufacturing plants.

    Parameters
    ----------
    method: str
        Name of the chosen AI method.
    """

    method = attr.ib()

    def __attrs_post_init__(self):
        """Attrs initialization."""
        self.method.register(self)

    def connection(self):
        """Connect function."""
        file_name = self.get_file_name_plant()
        self.connect = Com(file_name)
        return self.connect.connection()

    @abstractmethod
    def get_file_name_plant(self):
        """Name of the given plant file."""
        pass

    @abstractmethod
    def process_simulation(self):
        """Simulate in Tecnomatix."""
        pass

    @abstractmethod
    def update(self, data):
        """Update.

        Parameters
        ----------
        data:float
            Simulation data.
        """
        pass


@attr.s
class BasePlant(Plant):
    """A particularly adaptable plant.

    Parameters
    ----------
    method: str
        Name of the chosen AI method.
    v_i: list
        List of chosen input variables.
    v_o: list
        List of chosen output variables.
    filename: str
        Tecnomatix Plant Simulation complete file name (.spp)
    modelname: str
        Model frame name of the file, Default value="Model".
    """

    v_i = attr.ib()
    v_o = attr.ib()
    filename = attr.ib()
    modelname = attr.ib(default="Model")

    @v_i.validator
    def _validate_v_i(self, attribute, value):
        """Input value validator.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        if not isinstance(value, list):
            raise TypeError("v_i: Argument must be a list.")

    @v_o.validator
    def _validate_v_o(self, attribute, value):
        """Output value validator.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        if not isinstance(value, list):
            raise TypeError("v_o: Argument must be a list.")

    @filename.validator
    def _validate_filename(self, attribute, value):
        """File validator.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        if not isinstance(value, str):
            raise TypeError("File Name: Argument must be a string.")

    @modelname.validator
    def _validate_modelname(self, attribute, value):
        """Model validator.

        Parameters
        ----------
        value:int
            User-selected value.
        """
        if not isinstance(value, str):
            raise TypeError("Model Name: Argument must be a string.")

    def get_file_name_plant(self):
        """Get the name of the plant file.

        Return
        ------
        filename:str
            Model name.
        """
        return self.filename

    def update(self, data):
        """Update.

        Parameters
        ----------
        data:float
            Simulation data.

        Return
        -------
        r:float
            Reward value.
        """
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
        """Process simulation."""
        if self.connection():
            self.connect.setVisible(True)
            self.method.process()


# ============================================================================
# METHODS
# ============================================================================


@attr.s
class AutonomousDecisionSystem(metaclass=ABCMeta):
    """Autonomous decision system class."""

    method = attr.ib(init=False)

    def __attrs_post_init__(self):
        """Attrs initialization."""
        self.method = ""

    def register(self, who):
        """Subscribe registration.

        Parameters
        ----------
        who:str
            Node to subscribe.
        """
        self.subscriber = who

    @abstractmethod
    def process(self):
        """Process."""
        pass


@attr.s
class Qlearning(AutonomousDecisionSystem):
    """Implementation of the artificial intelligence method Q-Learning.

    Whose purpose is to obtain the optimal parameters from the trial and error
    method, in which it is penalized if the goal is not reached and is
    rewarded if it is reached, requiring for this a number of episodes.
    The Q table has a maximum of 625 rows, that is, up to 625 states are
    supported. These states are made up of 1 to 4 variables of the Tecnomatix
    Plant Simulation.
    Actions also depend on the chosen variables and their steps.
    The reward function depends on the results defined in the respective plant
    class.

    Parameters
    ----------
    v_i: list
        List of chosen input variables.
    episodes_max: positive int
        Total number of episodes to run. Should be a positive integer.
    steps_max: positive int
        Total number of steps in each episode. Should be a positive integer.
    alfa: float
        Reinforcement learning hyperparameter,
        learning rate, varies from 0 to 1. Default value= 0.10
    gamma: float
        Reinforcement learning hyperparameter,
        discount factor, varies from 0 to 1. Default value= 0.90
    epsilon: float
        Reinforcement learning hyperparameter,
        probability for the epsilon-greedy action selection,
        varies from 0 to 1. Default value= 0.10
    seed: int
        Seed value for the seed() method. Default value=None.
    """

    v_i = attr.ib()

    # number of episodes
    episodes_max = attr.ib()

    # number of steps
    steps_max = attr.ib()

    # initialize reward per episode
    r_episode = attr.ib(init=False)

    # reinforcement learning parameters
    alfa = attr.ib(default=0.10)
    gamma = attr.ib(default=0.90)
    epsilon = attr.ib(default=0.10)

    s = attr.ib(factory=list)
    a = attr.ib(factory=list)
    seed = attr.ib(default=None)

    def __attrs_post_init__(self):
        """Attrs initialization."""
        self.r_episode = np.arange(self.episodes_max, dtype=float)

        self._random = np.random.RandomState(seed=self.seed)

    @v_i.validator
    def _validate_v_i(self, attribute, value):
        """Initialize value validator."""
        if not isinstance(value, list):
            raise TypeError("v_i: Argument must be a list.")

    @episodes_max.validator
    def _validate_episodes_max(self, attribute, value):
        """Maximum epsilon validator."""
        if not isinstance(value, int):
            raise TypeError("Episodes Max: Argument must be an integer.")
        if value < 0:
            raise ValueError("Episodes Max: Argument must be higher than 0.")

    @steps_max.validator
    def _validate_steps_max(self, attribute, value):
        """Maximum steps validator."""
        if not isinstance(value, int):
            raise TypeError("Steps Max: Argument must be an integer.")
        if value < 0:
            raise ValueError("Steps Max: Argument must be higher than 0.")

    @alfa.validator
    def _validate_alfa(self, attribute, value):
        """Alpha validator."""
        if not isinstance(value, float):
            raise TypeError("Alfa: Argument must be a float.")
        if value < 0:
            raise ValueError("Alfa: Argument must be higher than 0.")
        if value > 1:
            raise ValueError("Alfa: Argument must be lower than 1.")

    @gamma.validator
    def _validate_gamma(self, attribute, value):
        """Gamma validator."""
        if not isinstance(value, float):
            raise TypeError("Gamma: Argument must be a float.")
        if value < 0:
            raise ValueError("Gamma: Argument must be higher than 0.")
        if value > 1:
            raise ValueError("Gamma: Argument must be lower than 1.")

    @epsilon.validator
    def _validate_epsilon(self, attribute, value):
        """Epsilon validator."""
        if not isinstance(value, float):
            raise TypeError("Epsilon: Argument must be a float.")
        if value < 0:
            raise ValueError("Epsilon: Argument must be higher than 0.")
        if value > 1:
            raise ValueError("Epsilon: Argument must be lower than 1.")

    def arrays(self):
        """Arrays for states and actions."""
        for idx, x in enumerate(self.v_i):
            self.s_idx = np.arange(
                x.lower_limit, x.upper_limit + x.step, x.step)
            self.a_idx = np.array([-x.step, 0, x.step])
            self.s.append(self.s_idx)
            self.a.append(self.a_idx)

    def ini_saq(self):
        """Initialize states, actions and Q table."""
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

    def choose_action(self, row):
        """Choose the action to follow.

        Parameters
        ----------
        row:int
            Number of rows.
        Return
        ------
        i:int
            Selected row.
        """
        p = self._random.random()
        if p < (1 - self.epsilon):
            i = np.argmax(self.Q[row, :])
        else:
            i = self._random.choice(self.actions.shape[0])
        return i

    def process(self):
        """Learning algorithms.

        Return
        ------
        r_episode:float
            Episode reward
        """
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

    __process = process


@attr.s
class Sarsa(Qlearning):
    """Implementation of the artificial intelligence method Sarsa.

    Whose purpose is to obtain the optimal parameters from the trial and error
    method, in which it is penalized if the goal is not reached and is
    rewarded if it is reached, requiring for this a number of episodes.
    The Q table has a maximum of 625 rows, that is, up to 625 states are
    supported. These states are made up of 1 to 4 variables of the Tecnomatix
    Plant Simulation.
    Actions also depend on the chosen variables and their steps.
    The reward function depends on the results defined in the respective plant
    class.

    Parameters
    ----------
    v_i: list
        List of chosen input variables.
    episodes_max: positive int
        Total number of episodes to run. Should be a positive integer.
    steps_max: positive int
        Total number of steps in each episode. Should be a positive integer.
    alfa: float
        Reinforcement learning hyperparameter,
        learning rate, varies from 0 to 1. Default value= 0.10
    gamma: float
        Reinforcement learning hyperparameter,
        discount factor, varies from 0 to 1. Default value= 0.90
    epsilon: float
        Reinforcement learning hyperparameter,
        probability for the epsilon-greedy action selection,
        varies from 0 to 1. Default value= 0.10
    seed: int
        Seed value for the seed() method. Default value=None.
    """

    def process(self):
        """Learning algorithm.

        Return
        ------
        r_episode:float
            Episode reward
        """
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
