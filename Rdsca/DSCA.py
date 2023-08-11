
# ============================================================================
# IMPORTS
# ============================================================================

import csv
import attr
import time
import random
import itertools
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
            self.connect.setvisible(True)
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
            self.connect.setvisible(True)
            route, Convergence_curve = self.method.DSCA(r)
            self.connect.closemodel()
            self.connect.quit()
        return route, Convergence_curve


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

    def DSCA(self, r, SearchAgents_n=6, Max_iter=50, prob=0.8):
        # Discrete Sine-Cosine Algorithm (Shubham Gupta)
        # X(t+1,inew) = Xti ⊕ (Cti ⊗ (Xtd ⊖ Xti))

        # Destination position
        D_pos = []
        D_score = float("inf")
        Convergence_curve = np.zeros(Max_iter)

        # Initialize the positions of search agents randomly
        Pos = []
        for z in range(SearchAgents_n):
            q = random.sample(r, len(r))
            Pos.append(q)

        # Main loop
        for m in range(Max_iter):
            # Evaluate the fitness of each search agent
            for i in range(SearchAgents_n):
                self.route = Pos[i]
                c, e = self.form_state()
                print("Test route", c, e)
                self.subscriber.update(c, e)
                res = self.subscriber.get_result()
                print("Result", res)
                fitness = res
                # Update Dest_Score (Best solution)
                if fitness < D_score:
                    D_score = fitness
                    D_pos = Pos[i].copy()
                    print("Best solution", D_score)

            # Update the Position of search agents
            for i in range(SearchAgents_n):
                r4 = random.random()
                if r4 > prob:
                    Pos[i] = random.sample(r, len(r))
                    print("Random solution", Pos[i])
                else:
                    r1 = m / Max_iter
                    # Update r2 and r3
                    r2 = (2 * np.pi) * random.random()
                    r3 = random.random()
                    print("r1", r1, "r2", r2, "r3", r3)
                    # Calculate number of features
                    F = self.n_features(Pos[i], D_pos)
                    print("F", F)
                    if r3 > 0.5:
                        nf = abs(r1 * np.sin(r2)) * F
                    else:
                        nf = abs(r1 * np.cos(r2)) * F
                    print("nf", nf)
                    if round(nf) > 0:
                        Pos[i] = self.update_sol(Pos[i], D_pos, nf)
                    print(Pos[i])

            Convergence_curve[m] = D_score

            if m % 1 == 0:
                print(["At iteration " + str(m) + " the best fitness is "
                      + str(D_score)])

        return D_pos, Convergence_curve


# ============================================================================
# MAIN
# ============================================================================


def plant_simulation_node():
    my_filename = "EjemploGola.spp"
    method = MethodB()
    plant = MH_opt(method=method, filename=my_filename)
     return plant.process_simulation(
         ["B21", "B71", "B91", "B141", "B191"])

if __name__ == '__main__':
    final_route, convergence_c = plant_simulation_node()


# ============================================================================
# RESULTS
# ============================================================================

plt.plot(convergence_c)
plt.title('Curva de convergencia r1')
plt.xlabel('Numero de Iteraciones')
plt.ylabel('Distancia recorrida (m)')
plt.show()

with open('prueba.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(convergence_c)
    writer.writerow(final_route)
