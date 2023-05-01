import rospy
import datetime
import argparse
import numpy as np
import time
import os
import sys
import traceback


print(sys.path)
print()
print(os.getcwd())


try:    
    print("Importing libs...")
    from src.stanford_lib.FollowController import FollowController
    from src.djipupper.Kinematics import four_legs_inverse_kinematics
    from src.djipupper.Config import Configuration
    from src.djipupper.IndividualConfig import SERIAL_PORT
    from src.djipupper import HardwareInterface
    from src.stanford_lib.State import State    
    from src.stanford_lib.Command import Command
except ImportError as e:
    print("Error importing libs...", traceback.format_exc())

from pupper_object_detection.msg import Detection, Detections, Test

DIRECTORY = "/home/pi/gmu-pupper/logs/"
FILE_DESCRIPTOR = "walking"

class PupperController:
    def __init__(self) -> None:
        self.log = rospy.get_param("/pupper_controller_node/log", False)
        self.zero = rospy.get_param("/pupper_controller_node/zero", False)
        self.home = rospy.get_param("/pupper_controller_node/home", True)

        rospy.loginfo(f"log = {self.log}\tzero = {self.zero}\thome= {self.home}")
        config = Configuration()
        self.config = Configuration()
        hardware_interface = HardwareInterface.HardwareInterface(port=SERIAL_PORT)
        time.sleep(0.1)

        # Create controller and user input handles
        controller = FollowController(config, four_legs_inverse_kinematics, automated=True)
        state = State(height=config.default_z_ref)
    
        self.config = config
        self.hardware_interface = hardware_interface
        self.state = state
        self.controller = controller

        self.summarize_config(config)
        rospy.init_node("PupperController")
        self.subscriber = rospy.Subscriber(
            "pupper_detections", Detections, self.handleNewDetections
        )

    def handleNewDetections(self, detectionsMsg: Detections):
        # rospy.loginfo(f"Received {len(detectionsMsg.detections)} detections")
        detection = (
            detectionsMsg.detections[0] if len(detectionsMsg.detections) > 0 else None
        )        
        self.controller.updateTarget(detection)
        

    def start(self):
        if self.log:
            today_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(
                DIRECTORY, FILE_DESCRIPTOR + "_" + today_string + ".csv"
            )
            log_file = open(filename, "w")
            self.hardware_interface.write_logfile_header(log_file)

        if self.zero:
            self.hardware_interface.set_joint_space_parameters(0, 4.0, 4.0)
            self.hardware_interface.set_actuator_postions(np.zeros((3, 4)))
            input(
                "Do you REALLY want to calibrate? Press enter to continue or ctrl-c to quit."
            )
            rospy.loginfo(f"Zeroing motors...", end="")
            self.hardware_interface.zero_motors()
            self.hardware_interface.set_max_current_from_file()
            rospy.loginfo(f"Done.")
        else:
            rospy.loginfo(f"Not zeroing motors!")

        rospy.loginfo("Calling deactivate")
        self.hardware_interface.deactivate()
        time.sleep(2)
        rospy.loginfo(f"Homing motors...")
        self.hardware_interface.home_motors()
        time.sleep(5)
        rospy.loginfo(f"Done setting up motors.")
        
        

        last_loop = time.time()
        
        self.state.activate_event = 1
        self.state.activation = 1
        self.state.deactivate_event = 0
        
        while not rospy.is_shutdown():
            command = Command(height=self.config.default_z_ref)

            state = self.state
            if state.activation == 0:
                time.sleep(0.02)                    
                if state.activate_event == 1:
                    rospy.loginfo(f"Robot activated.")
                    time.sleep(0.1)
                    self.hardware_interface.serial_handle.reset_input_buffer()
                    time.sleep(0.1)
                    rospy.loginfo("Telling pupper to stand")
                    self.hardware_interface.activate()
                    rospy.loginfo("Giving it some time to settle....")
                    time.sleep(5.0)
                    rospy.loginfo("Done standing")
                    state.activation = 1
                    continue
            elif state.activation == 1:
                now = time.time()                    
                if now - last_loop >= self.config.dt:                        
                    if state.deactivate_event == 1:
                        rospy.loginfo(f"Deactivating Robot")
                        rospy.loginfo(f"Waiting for L1 to activate robot.")
                        time.sleep(0.1)
                        self.hardware_interface.deactivate()
                        time.sleep(0.1)
                        state.activation = 0
                        break
                    self.controller.run(state, command)
                    self.hardware_interface.set_cartesian_positions(
                        state.final_foot_locations
                    )
                    last_loop = now        

        rospy.loginfo(f"Deactivating Robot")            
        self.controller.stop()
        time.sleep(0.1)
        command = Command(height=self.config.default_z_ref)
        command.stand_event = True
        self.controller.run(state, command)                
        time.sleep(0.1)
        self.hardware_interface.deactivate()
        rospy.loginfo("Exiting")
        

    def summarize_config(self, config):
        config = self.config
        # Print summary of configuration to console for tuning purposes
        rospy.loginfo(f"Summary of gait parameters:")
        rospy.loginfo(f"overlap time: ", config.overlap_time)
        rospy.loginfo(f"swing time: ", config.swing_time)
        rospy.loginfo(f"z clearance: ", config.z_clearance)
        rospy.loginfo(f"default height: ", config.default_z_ref)
        rospy.loginfo(f"x shift: ", config.x_shift)


if __name__ == "__main__":
    pupperController = PupperController()
    pupperController.start()
