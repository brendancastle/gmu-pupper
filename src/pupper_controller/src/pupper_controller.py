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
    from src.stanford_lib.JoystickInterface import JoystickInterface
    from src.stanford_lib.FollowController import FollowController
except ImportError as e:
    print("Error importing libs...", traceback.format_exc())

from pupper_object_detectison.msg import Detection, Detections, Test

DIRECTORY = "logs/"
FILE_DESCRIPTOR = "walking"


class PupperController:
    def __init__(self) -> None:
        self.log = rospy.get_param("/pupper_controller_node/log")
        self.zero = rospy.get_param("/pupper_controller_node/zero")
        self.home = rospy.get_param("/pupper_controller_node/home")
        rospy.loginfo(f"log = {self.log}\tzero = {self.zero}\thome= {self.home}")
        config = Configuration()
        hardware_interface = HardwareInterface.HardwareInterface(port=SERIAL_PORT)
        time.sleep(0.1)

        # Create controller and user input handles
        controller = FollowController(config, four_legs_inverse_kinematics)
        state = State(height=config.default_z_ref)
        rospy.loginfo("Creating joystick listener...", end="")
        joystick_interface = JoystickInterface(config)
        rospy.loginfo(f"Done.")

        self.config = config
        self.hardware_interface = hardware_interface
        self.state = state
        self.joystick_interface = joystick_interface
        self.controller = controller

        self.summarize_config(config)
        rospy.init_node("PupperController")
        self.subscriber = rospy.Subscriber(
            "pupper_detections", Detections, self.handleNewDetections
        )

    def handleNewDetections(self, detectionsMsg: Detections):
        rospy.logdebug(f"Received {len(detectionsMsg.detections)}")
        detection = (
            detectionsMsg.detections[0] if len(detectionsMsg.detections) > 0 else None
        )
        if detection is not None:
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

        if self.home:
            rospy.loginfo(f"Homing motors...", end="", flush=True)
            self.hardware_interface.home_motors()
            time.sleep(5)
            rospy.loginfo(f"Done.")

        rospy.loginfo(f"Waiting for L1 to activate robot.")

        last_loop = time.time()
        try:
            while True:
                state = self.state
                if state.activation == 0:
                    time.sleep(0.02)
                    self.set_color(self.config.ps4_deactivated_color)
                    command = self.joystick_interface.get_command(state)
                    if command.activate_event == 1:
                        rospy.loginfo(f"Robot activated.")
                        self.joystick_interface.set_color(self.config.ps4_color)
                        time.sleep(0.1)
                        self.hardware_interface.serial_handle.reset_input_buffer()
                        time.sleep(0.1)
                        self.ardware_interface.activate()
                        time.sleep(0.1)
                        state.activation = 1
                        continue
                elif state.activation == 1:
                    now = time.time()
                    if FLAGS.log:
                        any_data = self.hardware_interface.log_incoming_data(log_file)
                        if any_data:
                            rospy.loginfo(any_data["ts"])
                    if now - last_loop >= self.config.dt:
                        command = self.joystick_interface.get_command(state)
                        if command.deactivate_event == 1:
                            rospy.loginfo(f"Deactivating Robot")
                            rospy.loginfo(f"Waiting for L1 to activate robot.")
                            time.sleep(0.1)
                            self.hardware_interface.deactivate()
                            time.sleep(0.1)
                            state.activation = 0
                            continue
                        self.controller.run(state, command)
                        self.hardware_interface.set_cartesian_positions(
                            state.final_foot_locations
                        )
                        last_loop = now
        except KeyboardInterrupt:
            if self.log:
                rospy.loginfo(f"Closing log file")
                log_file.close()

    def summarize_config(self):
        config = self.config
        # Print summary of configuration to console for tuning purposes
        rospy.loginfo(f"Summary of gait parameters:")
        rospy.loginfo(f"overlap time: ", config.overlap_time)
        rospy.loginfo(f"swing time: ", config.swing_time)
        rospy.loginfo(f"z clearance: ", config.z_clearance)
        rospy.loginfo(f"default height: ", config.default_z_ref)
        rospy.loginfo(f"x shift: ", config.x_shift)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--zero", help="zero the motors", action="store_true")
    # parser.add_argument("--log", help="log pupper data to file", action="store_true")
    # parser.add_argument(
    #     "--home", help="home the motors (moves the legs)", action="store_true"
    # )
    # FLAGS = parser.parse_args()
    pupperController = PupperController()
    pupperController.start()
