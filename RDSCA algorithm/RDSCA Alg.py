
# ============================================================================
# IMPORTS
# ============================================================================

import attr
import time
import random
import itertools
import csv
import numpy as np
import matplotlib.pyplot as plt
from plant import Plant, AutonomousDecisionSystem

# ============================================================================
# PROBLEM DATA
# ============================================================================

buffers = ["B11", "B21", "B31", "B41", "B51", "B61", "B71", "B81", "B91",
           "B101", "B111", "B121", "B131", "B141", "B151", "B161", "B171",
           "B181", "B191", "B201", "B12", "B22", "B32", "B42", "B52", "B62",
           "B72", "B82",  "B92", "B102", "B112", "B122", "B132", "B142",
           "B152", "B162", "B172", "B182", "B192", "B202"]  # loading /
# unloading stations in order
station_loads = [50, 60, 20, 20, 10, 40, 40, 30, 20, 10, 20, 50, 30, 10, 20,
                 30, 10, 50, 25, 25, 50, 60, 20, 20, 10, 40, 40, 30, 20, 10,
                 20, 50, 30, 10, 20, 30, 10, 50, 25, 25]  # loading /
# unloading stations in order
times_machining = [80, 70, 200, 240, 400, 100, 100, 130, 200, 400, 200, 100,
                   125, 400, 200, 220, 400, 80, 140, 140]  # unit machining
# times in seconds of each station

# ============================================================================
# PLANTS
# ============================================================================


@attr.s
class Material_Handling(Plant):
    """Global simulation of 8 hours duration.

    """

    filename = attr.ib()

    def get_file_name_plant(self):
        return self.filename

    def update(self, c, state):
        if self.connect.getvalue(
                ".Models.Model3.EventController.SimTime") <= 28800:
            f = len(state)
            k = 0
            self.connect.setvalue(".Models.Model3.cargainicial", c)
            for e in range(1, int(f/2+1)):
                self.connect.setvalue(".Models.Model3.Rutas[1,%s]" % (e),
                                      state[k])
                self.connect.setvalue(".Models.Model3.Rutas[2,%s]" % (e),
                                      state[k + int(f/2)])
                k += 1
            self.connect.setvalue(".Models.Model3.Rutas[1,%s]" % str(f/2+1),
                                  "Source")

            self.connect.startsimulation(".Models.Model3")
            while self.connect.is_simulation_running():
                time.sleep(0.1)
        else:
            print("The simulation time is over")
            self.connect.closemodel()
            self.connect.quit()

    def get_result(self):
        a = np.zeros(40)
        b = 0
        if not self.connect.is_simulation_running():
            j = 0
            for i in buffers:
                a[j] = self.connect.getvalue(".Models.Model3.%s.NumMU" % i)
                j += 1
            b = self.connect.getvalue(
                ".Models.Model3.transporte.StatTraveledDistance")
        return a, b

    def requests(self, a):
        load = []
        unload = []
        for j in range(20):
            if (a[j] * times_machining[j]) <= 900:  # 15 min
                load.append(buffers[j])
        for k in range(20, 40):
            if a[k] >= station_loads[k]:
                unload.append(buffers[k])
        for x in load:
            for y in unload:
                if len(x) == 3 and len(y) == 3:
                    if x[1] == y[1]:
                        unload.remove(y)
                elif len(x) == 4 and len(y) == 4:
                    if x[2] == y[2]:
                        unload.remove(y)
        return load + unload

    def new_update(self, r, c, state):
        if len(r) > 0:
            if self.connect.getvalue(
                    ".Models.Model3.EventController.IsInitialized") is True:
                self.connect.setvalue(
                    ".Models.Model3.transporte.Stopped", False)
            self.update(c, state)
            print("r mayor a 0")
        else:
            self.connect.startsimulation(".Models.Model3")
            time.sleep(10)
            self.connect.stopsimulation()
            print("r menor a 0")
        r1, r2 = self.get_result()
        return r1, r2

    def process_simulation(self, r):
        if (self.connection()):
            self.connect.setvisible(False)
            route = self.method.process(r)
        return route

    def process_simulation2(self, r):
        route = self.method.process(r)
        return route

    __get_result = get_result
    __process_simulation = process_simulation


@attr.s
class MH_opt(Material_Handling):
    """Simulation to optimize each route (it is reset each time).

    """

    def get_result(self):
        b = 0
        if not self.connect.is_simulation_running():
            b = self.connect.getvalue(
                ".Models.Model3.transporte.StatTraveledDistance")
        self.connect.resetsimulation(".Models.Model3")
        return b

    def process_simulation(self, r):
        if (self.connection()):
            self.connect.setvisible(False)
            route, Convergence_curve, Q, r1_curves = self.method.RDSCA(r)
            self.connect.closemodel()
            self.connect.quit()
        return route, Convergence_curve, Q, r1_curves


# ============================================================================
# METHODS
# ============================================================================

@attr.s
class MethodA(AutonomousDecisionSystem):
    """For the Material_Handling class.

    """

    route = attr.ib(factory=list)

    def form_state(self):
        loads_1 = []
        loads_2 = []
        for i in range(len(self.route)):
            a = buffers.index(self.route[i])
            loads_1.append(station_loads[a])
            if a <= 19:
                loads_2.append(station_loads[a])
        c = sum(loads_2)  # initial load for Tecnomatix
        s = self.route + loads_1  # state for Tecnomatix
        return c, s

    def process(self, r):
        self.route = r
        c, s = self.form_state()
        print("Update ", c, s)
        r1, r2 = self.subscriber.new_update(self.route, c, s)
        print("Results", r1, r2)
        r3 = self.subscriber.requests(r1)
        print("Requests", r3)
        return r3

    __process = process


@attr.s
class MethodB(MethodA):
    """For the MH_opt class.

    """

    def test_routes(self, r):
        # Test all permutations
        perm = list(itertools.permutations(r))
        d = np.zeros(len(perm))
        st = []
        z = 0
        for i in range(len(perm)):
            self.route = list(perm[i])
            c, e = self.form_state()
            print("Test route", c, e)
            self.subscriber.update(c, e)
            res = self.subscriber.get_result()
            print("Result", res)
            d[z] = res
            st.append(self.route)
            z += 1
        return st[np.argmin(d)]

    def n_features(self, r, d):
        # Compare with the best solution and quantify differences
        F = 0
        for a in range(len(r)):
            if r[a] == d[a]:
                b = 0
            else:
                b = 1
            F += b
        return F

    def buscar_estado(self, estado):
        if estado <= 0.2:
            s = 0
        elif estado <= 0.4:
            s = 1
        elif estado <= 0.6:
            s = 2
        elif estado <= 0.8:
            s = 3
        else:
            s = 4
        return s

    def elegir_accion(self, fila, Q, A, epsilon):
        p = np.random.random()
        if p < (1 - epsilon):
            i = np.argmax(Q[fila, :])
        else:
            i = np.random.choice(len(A))
        return i

    def update_sol(self, r, d, nf):
        # Update the route with the best solution
        r_copy = []
        r_copy2 = []
        r_index = []
        # Copy the part to change
        for a in range(len(r)):
            if r[a] != d[a]:
                r_copy.append(r[a])
                r_index.append(r.index(r[a]))
        # Change
        idx = range(len(r_copy))
        if round(nf) < 3:
            i1, i2 = random.sample(idx, 2)
            r_copy[i1], r_copy[i2] = r_copy[i2], r_copy[i1]
        else:
            r_index2 = random.sample(idx, round(nf))
            for i in r_index2:
                r_copy2.append(r_copy[i])
            y = random.sample(r_copy2, len(r_copy2))
            for j in range(len(r_index2)):
                r_copy[r_index2[j]] = y[j]
        # Final update
        for p in range(len(r_copy)):
            r[r_index[p]] = r_copy[p]
        return r

    def RDSCA(self, r, Pop_size=6, Max_iter=30, prob=0.8):
        # Discrete Sine-Cosine Algorithm with Q-Learning for r1
        # X(t+1,inew) = Xti ⊕ (Cti ⊗ (Xtd ⊖ Xti))

        # Parametros Ql:
        alfa = 0.1
        gamma = 0.9
        epsilon = 1
        epsilon_minimo = 0.01
        epsilon_decay = 0.9

        # Initialize the Q-Tables
        Q1 = np.zeros((5, 4))
        Q2 = np.zeros((5, 4))
        Q3 = np.zeros((5, 4))
        Q4 = np.zeros((5, 4))
        Q5 = np.zeros((5, 4))
        Q6 = np.zeros((5, 4))
        Q = [Q1, Q2, Q3, Q4, Q5, Q6]

        # Destination position
        D_pos = []
        D_score = float("inf")
        Convergence_curve = np.zeros(Max_iter)

        # Initialize the positions of search agents randomly (solutions)
        Pos = []
        for z in range(Pop_size):
            q = random.sample(r, len(r))
            Pos.append(q)

        # Evaluate the fitness of each search agent at m=0
        X_pos = []
        X_fitness = np.zeros(Pop_size)
        for i in range(Pop_size):
            self.route = Pos[i]
            c, e = self.form_state()
            print("Test route", c, e)
            self.subscriber.update(c, e)
            res = self.subscriber.get_result()
            print("Result", res)
            fitness = res
            X_fitness[i] = fitness
            X_pos.append(Pos[i].copy())
            # Update Dest_Score (Best solution)
            if fitness < D_score:
                D_score = fitness
                D_pos = Pos[i].copy()
                print("Best solution", D_score)
        Convergence_curve[0] = D_score

        # Main loop
        r1_curves = np.zeros((Pop_size, Max_iter))
        for m in range(1, Max_iter):
            # Update the Position of search agents
            for i in range(Pop_size):
                r4 = random.random()
                if r4 > prob:
                    Pos[i] = random.sample(r, len(r))
                    print("Random solution", Pos[i])
                    # Evaluate the fitness
                    self.route = Pos[i]
                    c, e = self.form_state()
                    print("Test route", c, e)
                    self.subscriber.update(c, e)
                    res = self.subscriber.get_result()
                    print("Result", res)
                    fitness = res
                    # Update agent best solution
                    if fitness < X_fitness[i]:
                        X_fitness[i] = fitness
                        X_pos[i] = Pos[i].copy()
                else:
                    # State computation and Action execution
                    r1_1 = m / Max_iter
                    r1_2 = m / (Max_iter * 2)
                    r1_3 = 1 - np.exp(-(m) ** 0.5)
                    r1_4 = (1/(Max_iter * Max_iter)) * m * m
                    A = [r1_1, r1_2, r1_3, r1_4]
                    F = self.n_features(Pos[i], D_pos)
                    stateQ = F / len(D_pos)
                    print("F:", F, "StateQ:", stateQ)
                    s = self.buscar_estado(stateQ)
                    a = self.elegir_accion(s, Q[i], A, epsilon)
                    print("Estado:", s, "Accion:", a)

                    # Update r1
                    r1 = A[a]
                    r1_curves[i, m] = r1
                    # Update r2 and r3
                    r2 = (2 * np.pi) * random.random()
                    r3 = random.random()
                    print("r1", r1, "r2", r2, "r3", r3)

                    # Update positions
                    if r3 > 0.5:
                        nf = abs(r1 * np.sin(r2)) * F
                    else:
                        nf = abs(r1 * np.cos(r2)) * F
                    if round(nf) > 0:
                        Pos[i] = self.update_sol(Pos[i], D_pos, nf)
                    print(Pos[i])

                    # Evaluate the fitness of each search agent
                    self.route = Pos[i]
                    c, e = self.form_state()
                    print("Test route", c, e)
                    self.subscriber.update(c, e)
                    res = self.subscriber.get_result()
                    print("Result", res)
                    fitness = res
                    # Update agent best solution
                    if fitness < X_fitness[i]:
                        X_fitness[i] = fitness
                        X_pos[i] = Pos[i].copy()
                        reward = 1
                    else:
                        reward = -1
                    print("Reward", reward)

                    # Update state
                    F2 = self.n_features(Pos[i], D_pos)
                    stateQ2 = F2 / len(Pos[i])
                    s2 = self.buscar_estado(stateQ2)

                    # Update Q-Table
                    Q[i][s, a] = Q[i][s, a] + alfa * (reward + gamma * np.max(Q[i][s2, :]) - Q[i][s, a])

                # Update Dest_Score (Best solution)
                if fitness < D_score:
                    D_score = fitness
                    D_pos = Pos[i].copy()
                    print("Best solution", D_score)

            Convergence_curve[m] = D_score

            if epsilon > epsilon_minimo:
                epsilon = epsilon * epsilon_decay
                print("Epsilon", epsilon)

            if m % 1 == 0:
                print(["At iteration " + str(m) + " the best fitness is "
                      + str(D_score)])

        return D_pos, Convergence_curve, Q, r1_curves


# ============================================================================
# MAIN
# ============================================================================


def plant_simulation_node():
    my_filename = "EjemploGola.spp"
    method = MethodB()
    plant = MH_opt(method=method, filename=my_filename)
    return plant.process_simulation(["B11", "B41", "B61", "B151", "B161"])


if __name__ == '__main__':
    final_route, convergence_c, Q, r1_curves = plant_simulation_node()


# ============================================================================
# PLOTS and FILES
# ============================================================================

plt.plot(convergence_c)
plt.title('Curva de convergencia')
plt.xlabel('Numero de Iteraciones')
plt.ylabel('Distancia recorrida (m)')
plt.show()

with open('prueba1.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(convergence_c)
    writer.writerow(final_route)
    writer.writerow(Q[0])
    writer.writerow(Q[1])
    writer.writerow(Q[2])
    writer.writerow(Q[3])
    writer.writerow(Q[4])
    writer.writerow(Q[5])
