

from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


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
    supported. These states are made up of 1 to 5 variables of the Tecnomatix
    Plant Simulation.
    Actions also depend on the chosen variables and their steps.
    The reward function depends on the results defined in the respective plant
    class.
    """

    def __init__(self, alfa=0.10, gamma=0.90, epsilon=0.10, episodes_max=100,
                steps_max=100):
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

        # initialize actions
        self.actions = np.array([10, -10])

        # initialize states
        self.S = np.arange(60, 310, 10)

        # initialize Q table
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
