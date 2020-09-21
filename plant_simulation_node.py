

import rospy
import Material_Handling as M_A
import RL_Method_1 as RL


def plant_simulation_node():
    rospy.init_node('plant_simulation_node', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    method = RL.RL_Method_1()

    # plant = M_A.Material_Handling(method=None)
    plant = M_A.Material_Handling(method)
    plant.process_simulation()

    rate.sleep()


if __name__ == '__main__':
    try:
        plant_simulation_node()
    except rospy.ROSInterruptException:
        pass
