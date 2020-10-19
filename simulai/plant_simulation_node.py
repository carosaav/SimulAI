

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

    def __init__(self, method, filename, v_i, v_o):
        Plant.__init__(self, method)

        self.filename = filename
        self.v_i = v_i
        self.v_o = v_o

    def get_file_name_plant(self):
        return self.filename

    def update(self, data):
        for x in range(len(self.v_i)):
            self.connect.setValue(self.v_i[x].path, data[x])

        self.connect.startSimulation(".Models.Modelo")  # VER pasar path

        r = 0
        for k in range(len(self.v_o)):
            a_k = np.zeros(self.v_o[k].num_rows)
            for g in range(1, self.v_o[k].num_rows + 1):
                a_k[g - 1] = self.connect.getValue(
                    self.v_o[k].path[self.v_o[k].column, g])
            b_k = np.sum(a_k)
            r += b_k * (1 / len(self.v_o))

        self.connect.resetSimulation(".Models.Modelo")  # VER
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

    def __init__(self, v_i, alfa=0.10, gamma=0.90, epsilon=0.10,
                 episodes_max=100, steps_max=100):
        Autonomous_Decision_System.__init__(self)

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

    # initialize states, actions and Q table
    def ini_saq(self):
        for x in range(len(self.v_i)):
            self.s_x = np.arange(self.v_i[x].lower_limit,
                                 self.v_i[x].upperlimit + self.v_i[x].step,
                                 self.v_i[x].step)
            self.a_x = np.array([-self.v_i[x].step, 0, self.v_i[x].step])
            if len(self.v_i) == 1:
                self.S = np.array([self.s_0])
                self.actions = np.array([self.a_0])
            if len(self.v_i) == 2:
                a = np.repeat(self.s_0, self.s_1.shape[0])
                b = np.tile(self.s_1, self.s_0.shape[0])
                self.S = np.column_stack((a, b))
                c = np.repeat(self.a_0, self.a_1.shape[0])
                d = np.tile(self.a_1, self.a_0.shape[0])
                self.actions = np.column_stack((c, d))
            if len(self.v_i) == 3:
                a = np.repeat(self.s_0, self.s_1.shape[0] * self.s_2.shape[0])
                b = np.tile(self.s_1, self.s_0.shape[0] * self.s_2.shape[0])
                c = np.repeat(self.s_2, self.s_1.shape[0])
                d = np.tile(c, self.s_0.shape[0])
                e = np.column_stack((a, b))
                self.S = np.column_stack((e, d))
                f = np.repeat(self.a_0, self.a_1.shape[0] * self.a_2.shape[0])
                g = np.tile(self.a_1, self.a_0.shape[0] * self.a_2.shape[0])
                h = np.repeat(self.a_2, self.a_1.shape[0])
                i = np.tile(h, self.a_0.shape[0])
                j = np.column_stack((f, g))
                self.actions = np.column_stack((j, i))
            # if len(self.v_i) == 4:
            #     self.S =
        self.Q = np.zeros((self.S.shape[0], self.actions.shape[0]))

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


# PLANTS = {"Plant_1": Plant_1()}
# METHODS = {"Q_learning": Q_learning()}


def plant_simulation_node(pl):

    plant = pl
    plant.process_simulation()


# if __name__ == '__main__':
#     plant_simulation_node()
