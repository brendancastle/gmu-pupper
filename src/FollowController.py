from src.Controller import Controller
from src.State import BehaviorState
import numpy as np

from get_camera import Camera_serial

class FollowController(Controller):
    def __init__(self, config, inverse_kinematics):
        super().__init__(config, inverse_kinematics)

        self.in_follow_state = False
        self.following = False
        self.rx_ = 0.0
        self.ry_ = 0.0
        self.lx_ = 0.0
        self.ly_ = 0.0

        self.l_alpha = 0.15
        self.r_alpha = 0.1
        
        self.eps = 0.5
        # self.slow_down_distance = 1.0
        
        # probably from config file to get cam_skip_frames, distance 
        self.camera_only = False
        self.camera_module = Camera_serial(cam_skip_frames=100, distance=100, show_camera=self.camera_only)

    def depth_fn(self, d):
        if d > self.eps:
            return 1
        else:
            return 0

    def run(self, state, command):
        if command.follow_event:
            if self.in_follow_state == False:
                self.in_follow_state = True
                print("t pressed, entered follow state")
            else:
                self.in_follow_state = False
                command.stand_event = True
                super().run(state, command)
                print("t pressed, exited follow state")
        if self.in_follow_state:
            delta_yaw, depth = self.camera_module.get_camera_details()

            if self.camera_only == False and depth is not None:
                
                how_far = self.depth_fn(depth)
                
                if how_far != 0:  
                    self.ly_ = self.l_alpha * how_far + (1 - self.l_alpha) * self.ly_         # l_alpha*1 for forward. l_alpha*-1 for backward
                    x_vel = self.ly_ * self.config.max_x_velocity
                    y_vel = self.lx_ * -self.config.max_y_velocity
                    command.horizontal_velocity = np.array([x_vel, y_vel])

                    self.rx_ = self.r_alpha * delta_yaw + (1 - self.r_alpha) * self.rx_ #r_alpha*1 for right. r_alpha*-1 for left
                    command.yaw_rate = self.rx_ * self.config.max_yaw_rate
                    if state.behavior_state != BehaviorState.TROT:
                        command.trot_event = True

                    super().run(state, command)
                else:
                    if state.behavior_state != BehaviorState.REST:
                        print("goal reached")
                        command.stand_eveent = True
                        super().run(state, command)
        else:
            super().run(state, command)
