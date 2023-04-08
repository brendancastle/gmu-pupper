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
        self.slow_down_distance = 1.0
        
        # probably from config file to get cam_skip_frames, distance 
        self.camera_module = Camera_serial(cam_skip_frames=100, distance=100)

    
    def run(self, state, command):
        if command.follow_event:
            if self.in_follow_state == False:
                self.in_follow_state = True
                command.stand_event = True
                super().run(state, command)
                print("t pressed, in_follow_state entered")
            else: 
                self.in_follow_state = False
                super().run(state, command)
                print("t pressed, in_follow_state exited")
        elif self.in_follow_state:

            if command.start_stop_following_event and self.following == False:
                self.following = True
                self.count = 0
                print("u pressed, following started")
            elif command.start_stop_following_event and self.following == True:
                command.stand_event = True
                self.following = False
                super().run(state, command)
                print("u pressed, following exited")

            if self.following:
                delta_yaw, depth = self.camera_module.get_camera_details()
                
                

                def depth_fn(d):
                    # v1. slows down linearly between slow_down_distance and eps
                    # if d > self.slow_down_distance:
                    #     return 1
                    # elif d < self.slow_down_distance:
                    #     return (d-self.eps)/(self.slow_down_distance-self.eps)
                    # else:
                    #     return 0
                    # v2. stops at eps
                    if d > self.eps:
                        return 1
                    else:
                        return 0

                
                # self.lx_ = self.l_alpha * 0 + (1 - self.l_alpha) * self.lx_         # no straffing
                how_far = depth_fn(depth)
                
                self.ly_ = self.l_alpha * how_far + (1 - self.l_alpha) * self.ly_         # l_alpha*1 for forward. l_alpha*-1 for backward

                if how_far != 0:  
                    x_vel = self.ly_ * self.config.max_x_velocity
                    y_vel = self.lx_ * -self.config.max_y_velocity
                    command.horizontal_velocity = np.array([x_vel, y_vel])

                    self.rx_ = self.r_alpha * delta_yaw + (1 - self.r_alpha) * self.rx_ #r_alpha*1 for right. r_alpha*-1 for left
                    command.yaw_rate = self.rx_ * self.config.max_yaw_rate
                    if state.behavior_state != BehaviorState.TROT:
                        command.trot_event = True

                    super().run(state, command)
                else:
                    print("goal reached!")
                    command.stand_event = True
                    super().run(state, command)
                    
                
        else:
            super().run(state, command)
