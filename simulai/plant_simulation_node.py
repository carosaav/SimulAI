

# ============================================================================
# IMPORTS
# ============================================================================


from abc import ABCMeta, abstractmethod
from communication_interface import Communication_Interface as C_I
import numpy as np
import matplotlib.pyplot as plt


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
        self.connect = C_I(file_name)
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


class Plant_1(Plant):

    def __init__(self, method, filename):
        Plant.__init__(self, method)

        self.filename = filename

    def get_file_name_plant(self):
        return self.filename

    def update(self, data):
        self.connect.setValue(".Models.Modelo.espera", data)
        self.connect.startSimulation(".Models.Modelo")

        a = np.zeros(20)
        b = np.zeros(20)
        for h in range(1, 21):
            a[h-1] = self.connect.getValue(
                ".Models.Modelo.buffers[3,%s]" % (h))
            b[h-1] = self.connect.getValue(
                ".Models.Modelo.salidas[2,%s]" % (h))
        c = np.sum(a)
        d = np.sum(b)
        r = c*0.5+d*0.5

        self.connect.resetSimulation(".Models.Modelo")
        return r

    def process_simulation(self):
        if (self.connection()):
            self.connect.setVisible(True)
            self.method.process()
            self.method.plot()


# ============================================================================
# METHODS
# ============================================================================


class Autonomous_Decision_System(metaclass=ABCMeta):
    def __init__(self):
        self.method = ""

    def register(self, who):
        self.subscriber = who

    @abstractmethod
    def process(self):
        pass


class Q_learning(Autonomous_Decision_System):
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

    def __init__(self, alfa=0.10, gamma=0.90, epsilon=0.10, episodes_max=100,
                steps_max=100, *args):
        Autonomous_Decision_System.__init__(self)

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

        # save variables
        self.var = args

    # # initialize states REVISAR
    # def ini_states(self):
    #     for i, v in enumerate(self.var):
    #         s_i = np.arange({v}.lower_limit, {v}.upperlimit + {v}.step, {v}.step)
    #     if len(self.var) == 1:
    #         self.S = np.array([s_0])
    #     if len(self.var) == 2:
    #         a = np.repeat(s_0, s_1.shape[0])
    #         b = np.tile(s_1, s_0.shape[0])
    #         self.S = np.column_stack((a, b))
    #     if len(self.var) == 3:
    #         a = np.repeat(s_0, s_1.shape[0] * s_2.shape[0])
    #         b = np.tile(s_1, s_0.shape[0] * s_2.shape[0])
    #         c = np.repeat(s_3, s_2.shape[0])
    #         d = np.tile(c, s_0.shape[0])
    #         e = np.column_stack((a, b))
    #         self.S = np.column_stack((e, d))
    #     if len(self.var) == 4:
    #         self.S = 

    # # initialize actions and Q table REVISAR
    # def ini_actions_and_Q(self):
    #     for i, v in enumerate(self.var):
    #         a_i = np.array([-{v}.step, 0, {v}.step])
    #     if len(self.var) == 1:
    #         self.actions = np.array([a_0])
    #     if len(self.var) == 2:
    #         a = np.repeat(a_0, a_1.shape[0])
    #         b = np.tile(a_1, a_0.shape[0])
    #         self.actions = np.column_stack((a, b))
    #     if len(self.var) == 3:
    #         a = np.repeat(a_0, a_1.shape[0] * a_2.shape[0])
    #         b = np.tile(a_1, a_0.shape[0] * a_2.shape[0])
    #         c = np.repeat(a_3, a_2.shape[0])
    #         d = np.tile(c, a_0.shape[0])
    #         e = np.column_stack((a, b))
    #         self.actions = np.column_stack((e, d))
    #     if len(self.var) == 4:
    #         self.S = 

    #     # initialize Q table
    #     self.Q = np.zeros((self.S.shape[0], self.actions.shape[0]))

    # choose action
    def choose_action(self, row):
        p = np.random.random()
        if p < (1-self.epsilon):
            i = np.argmax(self.Q[row, :])
        else:
            i = np.random.choice(2)
        return (i)

    # reinforcement learning process
    def process(self):
        for n in range(self.episodes_max):
            S0 = self.S[0]
            t = 0
            r_acum = 0
            res0 = self.subscriber.update(S0)
            while t < self.steps_max:
                # find k index of current state
                for k in range(25):
                    if self.S[k] == S0:
                        break
                # choose action from row k
                j = self.choose_action(k)
                # update state
                Snew = S0 + self.actions[j]
                # limites
                if Snew > 300:
                    Snew -= 10
                elif Snew < 60:
                    Snew += 10
                # update simulation result
                res1 = self.subscriber.update(Snew)
                # reward
                if res1 < res0:
                    r = 1
                else:
                    r = 0
                # find index of new state
                for z in range(25):
                    if self.S[z] == Snew:
                        break
                # update Q table
                self.Q[k, j] = self.Q[k, j]
                + self.alfa * (r + self.gamma * np.max(self.Q[z, :]) - self.Q[k, j])
                # update parameters
                t += 1
                S0 = Snew
                res0 = res1
                r_acum = r_acum + r
                self.r_episode[n] = r_acum
        return self.r_episode

    def plot(self):
        plt.plot(self.r_episode, "b-")
        plt.axis([0, self.episodes_max, 0, self.steps_max])
        plt.title("Accumulated reward per episode")
        plt.xlabel("Number of episodes")
        plt.ylabel("Accumulated reward")
        plt.show()


# ============================================================================
# MAIN
# ============================================================================


PLANTS = {"Plant_1": Plant_1()}
METHODS = {"Q_learning": Q_learning()}


def plant_simulation_node(m, p, filename, *args):

    method = METHODS[m](args)
    plant = PLANTS[p](method, filename)
    plant.process_simulation()


if __name__ == '__main__':
    plant_simulation_node()
