import numpy as np
from enum import Enum


class State:
    def __init__(self, height):
        self.horizontal_velocity = np.array([0.0, 0.0])
        self.yaw_rate = 0.0
        self.height = height
        self.pitch = 0.0
        self.roll = 0.0
        self.activation = 0
        self.behavior_state = BehaviorState.REST
        self.deactivate_event = 0
        self.activate_event = 0

        self.ticks = 0
        self.foot_locations = np.zeros((3, 4))
        self.final_foot_locations = np.zeros((3, 4))
        self.joint_angles = np.zeros((3, 4))

        self.quat_orientation = np.array([1, 0, 0, 0])


class BehaviorState(Enum):
    DEACTIVATED = -1
    REST = 0
    TROT = 1
    HOP = 2
    FINISHHOP = 3
    WALK = 4
    FOLLOW = 5
