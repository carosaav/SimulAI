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
    """This class allows you to enter the chosen Tecnomatix Plant Simulation
    variables as input to the artificial intelligence method.
    Up to 4 discrete variables are allowed that must conform up to 625
    possible states. For example, if 4 variables are chosen, each of them
    can take 5 possible values and States= (Var1, Var2, Var3, Var4)
    """

    def __init__(self, name, lower_limit, upper_limit, step, path):
        self.name = name
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.step = step
        self.path = path


class OutcomeVariable:
    """This class allows you to enter the chosen Tecnomatix Plant Simulation
    variables as output to the artificial intelligence method.
    These variables must be of type Data Table.
    The chosen column from which to extract the results and the number of
    rows it has must be indicated.
    """

    def __init__(self, name, path, column, num_rows):
        self.name = name
        self.path = path
        self.column = column
        self.num_rows = num_rows


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


class Plant1(Plant):
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


class Qlearning(AutonomousDecisionSystem):
    """This class implements reinforcement learning using the well-known
    tabular Q-learning algorithm with a epsilon-greedy exploration strategy.
    The alpha, gamma and epsilon parameters are given by default, as well as
    the number of episodes and steps of the algorithm, what the user can adapt
    to his situation.
    The Q table has a maximum of 625 rows, that is, up to 625 states are
    supported. These states are made up of 1 to 4 variables of the Tecnomatix
    Plant Simulation.
    Actions also depend on the chosen variables and their steps.
    The reward function depends on the results defined in the respective plant
    class.
    """

    def __init__(
        self, v_i, alfa=0.10, gamma=0.90, epsilon=0.10,
        episodes_max=100, steps_max=100
    ):
        AutonomousDecisionSystem.__init__(self)

        self.v_i = v_i

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

    # arrays for states and actions
    def arrays(self):
        self.s = []
        self.a = []
        for idx, x in enumerate(self.v_i):
            self.s_idx = np.arange(
                x.lower_limit, x.upper_limit + x.step, x.step)
            self.a_idx = np.array([-x.step, 0, x.step])
            self.s.append(self.s_idx)
            self.a.append(self.a_idx)
        return self.s, self.a

    # initialize states, actions and Q table
    def ini_saq(self):
        self.arrays()
        if len(self.v_i) == 1:
            self.S = self.s[0]
            self.actions = self.a[0]
        if len(self.v_i) == 2:
            self.S = np.column_stack(
                (
                    np.repeat(self.s[0], self.s[1].shape[0]),
                    np.tile(self.s[1], self.s[0].shape[0]),
                )
            )
            self.actions = np.column_stack(
                (
                    np.repeat(self.a[0], self.a[1].shape[0]),
                    np.tile(self.a[1], self.a[0].shape[0]),
                )
            )
        if len(self.v_i) == 3:
            b = np.repeat(self.s[0], self.s[1].shape[0] * self.s[2].shape[0])
            c = np.tile(self.s[1], self.s[0].shape[0] * self.s[2].shape[0])
            d = np.repeat(self.s[2], self.s[1].shape[0])
            self.S = np.column_stack(
                (np.column_stack((b, c)), np.tile(d, self.s[0].shape[0]))
            )
            f = np.repeat(self.a[0], self.a[1].shape[0] * self.a[2].shape[0])
            g = np.tile(self.a[1], self.a[0].shape[0] * self.a[2].shape[0])
            h = np.repeat(self.a[2], self.a[1].shape[0])
            self.actions = np.column_stack(
                (np.column_stack((f, g)), np.tile(h, self.a[0].shape[0]))
            )
        # if len(self.v_i) == 4:
        self.Q = np.zeros((self.S.shape[0], self.actions.shape[0]))
        return self.S, self.actions, self.Q

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
                +self.alfa * (r + self.gamma * np.max(
                            self.Q[z, :]) - self.Q[k, j])
                # update parameters
                t += 1
                S0 = Snew
                res0 = res1
                r_acum = r_acum + r
                self.r_episode[n] = r_acum
        return self.r_episode


# ============================================================================
# MAIN
# ============================================================================


def simulation_node(pl):

    plant = pl
    plant.process_simulation()
