

from Plant import Plant
import numpy as np
import rospy
from std_msgs.msg import String


class Material_Handling(Plant):

    # def __init__(self, method=None):
    #     self.pub = rospy.Publisher('plant_simulation_topic', String, queue_size=10)
    #     Plant.__init__(self, method)

    def __init__(self, method):
        self.pub = rospy.Publisher('plant_simulation_topic', String, queue_size=10)
        Plant.__init__(self, method)

    def get_file_name_plant(self):
        return "MaterialHandling.spp"

    def update(self, data):
        self.connect.setValue(".Models.Modelo.espera", data)
        self.connect.startSimulation(".Models.Modelo")

        a = np.zeros(20)
        b = np.zeros(20)
        for h in range(1, 21):
            a[h-1] = self.connect.getValue(".Models.Modelo.buffers[3,%s]" % (h))
            b[h-1] = self.connect.getValue(".Models.Modelo.salidas[2,%s]" % (h))
        c = np.sum(a)
        d = np.sum(b)
        r = c*0.5+d*0.5

        self.connect.resetSimulation(".Models.Modelo")

        # topic publish
        resultado_str = "YES"
        rospy.loginfo(resultado_str)
        self.pub.publish(resultado_str)
        return (r)

    def process_simulation(self):
        if (self.connection()):
            self.connect.setVisible(True)
            self.method.process()
