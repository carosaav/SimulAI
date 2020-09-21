

from Autonomous_Decision_System import Autonomous_Decision_System
import numpy as np


class RL_Method_1(Autonomous_Decision_System):

    def __init__(self):
        Autonomous_Decision_System.__init__(self)

        # datos reinforcement learning
        self.alfa = 0.10
        self.gamma = 0.90
        self.epsilon = 0.10

        # numero de episodios
        self.ep_maximo = 5

        # numero de pasos
        self.t_maximo = 10

        # inicializar recompensa por episodio
        self.r_episodio = np.arange(self.ep_maximo, dtype=float)

        # inicializar acciones
        self.acciones = np.array([10, -10])

        # inicializar tabla Q
        self.Q = np.zeros((25, 2))

        # inicializar S
        self.S = np.arange(60, 310, 10)

    # funcion elegir accion
    def elegir_accion(self, fila):
        p = np.random.random()
        if p < (1-self.epsilon):
            i = np.argmax(self.Q[fila, :])
        else:
            i = np.random.choice(2)
        return (i)

    # funcion rl- actualizar estados y matriz Q
    def process(self):
        for n in range(self.ep_maximo):
            S0 = self.S[12]  # 3 minutos
            t = 0
            r_acum = 0
            while t < self.t_maximo:
                res0 = self.subscriber.update(S0)
                # buscar indice k del estado actual
                for k in range(25):
                    if self.S[k] == S0:
                        break
                # elegir accion de la fila k
                j = self.elegir_accion(k)
                # actualizar estado
                Snew = S0 + self.acciones[j]
                # limites
                if Snew > 300:
                    Snew -= 10
                elif Snew < 60:
                    Snew += 10
                print(Snew)
                # actualizar resultado simulacion
                res1 = self.subscriber.update(Snew)
                print(res1)
                # recompensa
                if res1 < res0:
                    r = 1
                else:
                    r = 0
                print(r)
                # buscar indice del estado nuevo S'
                for z in range(25):
                    if self.S[z] == Snew:
                        break
                # actualizar matriz Q
                self.Q[k, j] = self.Q[k, j] + self.alfa * (r + self.gamma * np.max(self.Q[z, :]) - self.Q[k, j])
                # actualizar parametros
                t += 1
                S0 = Snew
                r_acum = r_acum + r
                r_medio = r_acum/t
                self.r_episodio[n] = r_medio
