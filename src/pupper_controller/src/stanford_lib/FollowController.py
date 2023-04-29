from src.stanford_lib.Controller import Controller
from src.stanford_lib.State import BehaviorState
import numpy as np

from pupper_object_detection.msg import Detection

class FollowController(Controller):
    def __init__(self, config, inverse_kinematics, automated=True):
        super().__init__(config, inverse_kinematics)

        self.in_follow_state = False
        self.following = False
        self.rx_ = 0.0
        self.ry_ = 0.0
        self.lx_ = 0.0
        self.ly_ = 0.0

        self.l_alpha = 0.15
        self.r_alpha = 0.1
        
        self.config.max_yaw_rate = 0.05
        self.config.max_x_velocity = 0.1
        self.config.mx_y_velocity = 0.1
        self.eps = 0.2
        
        # self.slow_down_distance = 1.0
        
        # probably from config file to get cam_skip_frames, distance 
        self.camera_only = True 
        # cam_skip_frames =-1 for checking at every time step.

        self.goal = None
        self.automated = automated
        self.in_follow_state = automated
        self.centerPointOfTarget = None
        self.init_run = False

    def updateTarget(self, detection:Detection):
        self.depthOfTarget = detection.depthAtCenter
        x = detection.xmin + (detection.xmax - detection.xmin)/2
        y = detection.ymin + (detection.ymax - detection.ymin)/2        
        self.centerPointOfTarget = (x,y) 
        print(f"Set target to {(x,y)},{self.depthOfTarget}")

    def depth_fn(self, d):
        if d > self.eps:
            return 1
        else:
            return 0
        
    def yaw_from_coords(self, xy, depth): # v1 is that it simply turns right or left depending on where the object is. its not very fine tuned.
        self.pupper_center = 320 # assuming pupper is at center of camera. width of camera: 120
        x,_ = xy
        if x > self.pupper_center: # on right
            return 1
        elif x < self.pupper_center: # on lefr
            return -1
        else:
            return 0 

    def set_goal(self, object_center, depth):
        self.goal = (object_center, depth)

    def run(self, state, command):
        if not self.automated:
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
            object_center = None
            depth = None
            if self.centerPointOfTarget is not None:
                object_center, depth = self.centerPointOfTarget, self.depthOfTarget/1000.0  
            # print("object_center, depth:",object_center, depth)
            # print(f"goal_pixels: {object_center}, depth: {depth:.2f}")
            if self.goal is None:
                if state.behavior_state != BehaviorState.REST or not self.init_run:
                    command.stand_event = True
                    super().run(state, command)
                    print("pupper standing because no object detected and no goal is set already")
                    self.init_run=True # if this isnt called the pupper never stands if there is no goal 

            if object_center is not None and depth is not None:
                self.goal = (object_center, depth)
            
            if self.goal is not None:
                delta_yaw = self.yaw_from_coords(self.goal[0], self.goal[1])
                how_far = self.depth_fn(self.goal[1])
                
                if how_far != 0:  
                    self.ly_ = self.l_alpha * how_far + (1 - self.l_alpha) * self.ly_         # l_alpha*1 for forward. l_alpha*-1 for backward
                    x_vel = self.ly_ * self.config.max_x_velocity
                    y_vel = self.lx_ * -self.config.max_y_velocity
                    
                    command.horizontal_velocity = np.array([x_vel, y_vel])
        
                    self.rx_ = self.r_alpha * delta_yaw + (1 - self.r_alpha) * self.rx_ #r_alpha*1 for right. r_alpha*-1 for left
                     
                    command.yaw_rate = self.rx_ * -self.config.max_yaw_rate

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


    # some logic:
    # 1. so pupper goes in follow mode
    # 2. it starts detecting using camera
    # 3. while camera has no object --> it just stands
    # 4. when camera detects an object and returns depth, (x,y) --> saves depth and (x,y) and starts moving to it
    # 5. now that it has a goal, keep going towards goal and check camera every x units 
    #   - this is useful for simulation cuz loading camera takes a lot of compute
    #   - this may be useful irl cuz camera will be shaking up and down a lot
    #   - alternatively we can check camera more frequently if goal is near