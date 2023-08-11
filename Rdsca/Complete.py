
# ============================================================================
# IMPORTS
# ============================================================================

import attr
import time
import csv
import random
import numpy as np
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
    filename = attr.ib()

    def get_file_name_plant(self):
        return self.filename

    def update(self, c, state):
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

    def get_result(self):
        a = np.zeros(40)
        b = 0
        while self.connect.is_simulation_running():
            time.sleep(0.1)
        if not self.connect.is_simulation_running():
            j = 0
            for i in buffers:
                a[j] = self.connect.getvalue(".Models.Model3.%s.NumMU" % i)
                j += 1
            b = self.connect.getvalue(
                ".Models.Model3.transporte.StatTraveledDistance")
        return a, b

    def requests(self, a):
        load = {}
        unload = {}
        for j in range(20):
            if (a[j] * times_machining[j]) <= 900:  # 15 min
                load[buffers[j]] = a[j] * times_machining[j]
        for k in range(20, 40):
            if a[k] > station_loads[k]:
                unload[buffers[k]] = a[k] - station_loads[k]
        print(load)
        print(unload)
        # Prioridades: ordeno los diccionarios
        sorted_load = dict(sorted(load.items(), key=lambda item: item[1]))
        sorted_unload = dict(sorted(unload.items(), key=lambda item: item[1],
                             reverse=True))
        a = list(sorted_load.keys())
        b = list(sorted_unload.keys())
        for x in a:
            for y in b:
                if len(x) == 3 and len(y) == 3:
                    if x[1] == y[1]:
                        b.remove(y)
                elif len(x) == 4 and len(y) == 4:
                    if x[2] == y[2]:
                        b.remove(y)
        req = a + b
        print(req)
        # Limite: 10 estaciones
        # while len(req) > 10:
        #     req.pop()
        return req

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

    def final_results(self):
        dist = []
        waits = np.zeros(20)
        mu, d = self.get_result()
        dist.append(d)
        for j in range(20):
            waits[j] = self.connect.getvalue(
                ".Models.Model3.Estacion%s.statWaitingPortion" % (j+1))
        # self.connect.closemodel()
        # self.connect.quit()
        return mu, dist, waits

    def process_simulation(self, r):
        if (self.connection()):
            self.connect.setvisible(True)
            route = self.method.process(r)
        return route

    def process_simulation2(self, r):
        if self.connect.getvalue(
                ".Models.Model3.EventController.SimTime") <= 14400:
            cond = True
            route = self.method.process(r)
        else:
            cond = False
            route = []
            print("The simulation time is over")
        return route, cond


# ============================================================================
# METHODS
# ============================================================================

@attr.s
class MethodA(AutonomousDecisionSystem):
    route = attr.ib(factory=list)

    def form_state(self):
        loads_1 = []
        loads_2 = []
        for i in range(len(self.route)):
            a = buffers.index(self.route[i])
            loads_1.append(station_loads[a])
            if a <= 19:
                loads_2.append(station_loads[a])
        c = sum(loads_2)
        s = self.route + loads_1
        return c, s

    def ruta_orden(self, r):
        parte_a = []
        parte_b = []
        if len(r) > 0:
            for x in r:
                if len(x) == 3:
                    parte_a.append(x)
                else:
                    parte_b.append(x)
            parte_a.sort()
            parte_b.sort()
        return parte_a + parte_b

    def process(self, r):
        self.route = r
        c, s = self.form_state()
        print("Update ", c, s)
        r1, r2 = self.subscriber.new_update(self.route, c, s)
        print("Results", r1, r2)
        r3 = self.subscriber.requests(r1)
        print("Requests", r3)
        # ruta random:
        # r4 = random.sample(r3, len(r3))
        # ruta en orden:
        # r4 = self.ruta_orden(r3)
        # ruta con prioridades:
        r4 = r3
        print("Route", r4)
        return r4


# ============================================================================
# MAIN
# ============================================================================


def plant_simulation_node():
    filename = "model.spp"
    method = MethodA()
    plant = Material_Handling(method=method, filename=filename)

    r = plant.process_simulation(["B191"])
    run = 0
    condition = True

    while condition:
        print("Run", run)
        route, condition = plant.process_simulation2(r)
        r = route
        run += 1

    mu, dist, waits = plant.final_results()

    with open('complete.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(mu)
        writer.writerow(dist)
        writer.writerow(waits)


if __name__ == '__main__':
    plant_simulation_node()
