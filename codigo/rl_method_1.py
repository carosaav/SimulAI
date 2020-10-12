from Autonomous_Decision_System import Autonomous_Decision_System
import numpy as np

# Algoritmo Q-Learning

class RL_Method_1(Autonomous_Decision_System):

    def __init__(self):
        Autonomous_Decision_System.__init__(self)

        # Parámetros de reinforcement learning
        self.alfa = 0.10
        self.gamma = 0.90
        self.epsilon = 0.10

        # Iumero de episodios
        self.ep_maximo = 5

        # numero de pasos
        self.t_maximo = 10

        # Inicializar recompensa por episodio
        self.r_episodio = np.arange(self.ep_maximo, dtype=float)

        # Inicializar acciones
        self.acciones = np.array([10, -10])

        # Inicializar matriz Q
        self.Q = np.zeros((25, 2))

        # Inicializar S
        self.S = np.arange(60, 310, 10)

    # Elegir accion
    def elegir_accion(self, fila):
        p = np.random.random()
        if p < (1 - self.epsilon):
            i = np.argmax(self.Q[fila, :])
        else:
            i = np.random.choice(2)
        return (i)

    # Actualización de estados y de matriz Q
    def process(self):
        for n in range(self.ep_maximo):
            S0 = self.S[12]
            t = 0
            r_acum = 0
            while t < self.t_maximo:
                res0 = self.subscriber.update(S0)
                # Búsqueda estado actual en la matriz Q
                for k in range(25):
                    if self.S[k] == S0:
                        break
                # Elegir accion de la fila k
                j = self.elegir_accion(k)
                # Actualizar estado
                Snew = S0 + self.acciones[j]
                # Límites
                if Snew > 300:
                    Snew -= 10
                elif Snew < 60:
                    Snew += 10
                # Actualización del resultado simulacion
                res1 = self.subscriber.update(Snew)
                # Recompensa
                if res1 < res0:
                    r = 1
                else:
                    r = 0
                # Búsqueda del estado nuevo S'
                for z in range(25):
                    if self.S[z] == Snew:
                        break
                # Actualización de matriz Q
                self.Q[k, j] = self.Q[k, j]
                + self.alfa* (r + self.gamma * np.max(self.Q[z, :])
                 - self.Q[k, j])
                # Actualización de parametros de Reinforcement Learning
                t += 1
                S0 = Snew
                r_acum = r_acum + r
                r_medio = r_acum / t
                self.r_episodio[n] = r_medio

# Algoritmo Sarsa
class RL_Method_2(Autonomous_Decision_System):

    def __init__(self):
        Autonomous_Decision_System.__init__(self)

        # Parámetros de Reinforcement Learning
        self.alfa = 0.10
        self.gamma = 0.90
        self.epsilon = 0.10

        # Número de episodios
        self.ep_maximo = 5

        # Número de pasos
        self.t_maximo = 10

        # Inicializa la recompensa por episodio
        self.r_episodio = np.arange(self.ep_maximo, dtype=float)

        # Inicializa las acciones
        self.acciones = np.array([10, -10])

        # Inicializar matriz Q
        self.Q = np.zeros((25, 2))

        # inicializar S
        self.S = np.arange(60, 310, 10)

    # Elegi la  accion a seguir
    def elegir_accion(self, fila):
        p = np.random.random()
        if p < (1 - self.epsilon):
            i = np.argmax(self.Q[fila, :])
        else:
            i = np.random.choice(2)
        return (i)

    # Actualizar estados y matriz Q
    def process(self):
        for n in range(self.ep_maximo):
            S0 = self.S[12]
            t = 0
            r_acum = 0
            while t < self.t_maximo:
                res0 = self.subscriber.update(S0)
                # Buscar el estado actual
                for k in range(25):
                    if self.S[k] == S0:
                        break
                # Elegir accion de la fila k
                j_anterior = j
                j = self.elegir_accion(k)
                # Actualizar estado
                Snew = S0 + self.acciones[j]
                # Límites del cálculo
                if Snew > 300:
                    Snew -= 10
                elif Snew < 60:
                    Snew += 10
                # Actualiza el resultado simulacion
                res1 = self.subscriber.update(Snew)
                # Recompensa
                if res1 < res0:
                    r = 1
                else:
                    r = 0
                # Buscar el estado nuevo S'
                for z in range(25):
                    if self.S[z] == Snew:
                        break
                # Actualizar matriz Q
                self.Q[k, j] = self.Q[k, j]
                + self.alfa * (r + self.gamma * (self.Q[z, j]) 
                - self.Q[k, j_anterior])
                # Actualización de parametros de Reinforcement Learning
                t += 1
                S0 = Snew
                r_acum = r_acum + r
                r_medio = r_acum / t
                self.r_episodio[n] = r_medio
