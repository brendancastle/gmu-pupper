import threading
from src.stanford_lib.Controller import Controller
from src.stanford_lib.State import BehaviorState
import numpy as np

from pupper_object_detection.msg import Detection

class FollowController(Controller):
    def __init__(self, config, inverse_kinematics, automated=True, useCenterPointSmoothing=False):
        super().__init__(config, inverse_kinematics)

        # automation
        self.automated = automated
        self.in_follow_state = automated

        # movement
        self.rx_ = 0.0
        self.ry_ = 0.0
        self.lx_ = 0.0
        self.ly_ = 0.0

        self.l_alpha = 0.15
        self.r_alpha = 0.1
        
        # self.config.max_yaw_rate = 1.0
        # self.config.max_x_velocity = 0.6
        # self.config.max_y_velocity = 0.6

        # distance
        self.stopDistance = self.config.stop_distance
        
        # object detection
        self.goal = None
        self.centerPointOfTarget = None
        self.pupper_center = 320 # assuming pupper is at center of camera. width of camera: 120
        self.camera_width = 640
        self.sensingzone_args = {
            "min_edge_zone": 40, 
            "max_edge_zone": 300, 
            "max_depth":2, 
            "min_depth":1
            }
        self.useCenterPointSmoothing = useCenterPointSmoothing
        self.centerPointAlpha = 0.15
        self.goalLock = threading.Lock()

        # finding object after no detection
        self.lastGoal = None
        self.middle_pixel_threshold = 160
        self.reorient_state = False
        
        
    def updateTarget(self, detection:Detection):
        with self.goalLock:
            if detection is not None:    
                self.depthOfTarget = detection.depthAtCenter/1000.0

                x = detection.xmin + (detection.xmax - detection.xmin)/2    
                y = detection.ymin + (detection.ymax - detection.ymin)/2
                
                if self.useCenterPointSmoothing and self.centerPointOfTarget is not None:
                    x = self.centerPointOfTarget[0] * self.centerPointAlpha + (1-self.centerPointAlpha) * x
                    y = self.centerPointOfTarget[1] * self.centerPointAlpha + (1-self.centerPointAlpha) * y
                
                self.centerPointOfTarget = (x,y) 
                print(f"Set target to {(x,y)},{self.depthOfTarget}")
                self.lastGoal = ((x,y), self.depthOfTarget)
                self.set_goal((x,y), self.depthOfTarget)
            else:
                print("Clearing goal")
                self.goal = None


    def depth_fn(self, d):
        if d > self.stopDistance:
            return 1
        else:
            return 0

    def stop(self):
        self.in_follow_state = False
        
    def yaw_from_coords(self, xy, depth, use_sensingzone = True): # v1 is that it simply turns right or left depending on where the object is. its not very fine tuned.
        x,_ = xy
        if use_sensingzone:
            if depth > self.sensingzone_args["max_depth"]:
                depth = self.sensingzone_args["max_depth"]
            elif depth < self.sensingzone_args["min_depth"]:
                depth = self.sensingzone_args["min_depth"]
            else:
                depth = depth
            
            scale = (depth-self.sensingzone_args["min_depth"])/(self.sensingzone_args["max_depth"]-self.sensingzone_args["min_depth"])
            
            edge = self.sensingzone_args["max_edge_zone"] - (self.sensingzone_args["max_edge_zone"]-self.sensingzone_args["min_edge_zone"])*scale
            
            if x > self.camera_width - edge: #this is assuming pupper is at the center of camera
                return 1
            elif x < edge:
                return -1
            else:
                return 0
        else:
            if x > self.pupper_center: # on right
                return 1
            elif x < self.pupper_center: # on lefr
                return -1
            else:
                return 0 

    def set_goal(self, object_center, depth):
        self.goal = (object_center, depth)
        

    def run(self, state, command):
        if self.in_follow_state:                        
                        
            with self.goalLock:
                goal = self.goal

            if goal is None:
                if state.behavior_state != BehaviorState.REST:
                    command.stand_event = True
                    super().run(state, command)
                    print("pupper standing because no object detected and no goal is set already")
                    # v2 self reorient:
                    self.ly_ = 0
                    self.rx_ = 0
                    self.reorient_state = True

            if self.reorient_state: # assumes rest state
                lower_pixel_bound = self.camera_width//2 - self.middle_pixel_threshold
                upper_pixel_bound = self.camera_width//2 +self.middle_pixel_threshold

                if goal is None or not (goal[0][0] > lower_pixel_bound and goal[0][0] < upper_pixel_bound): 
                    # v1: keep turning until goal found --> failed because turning while standing doesnt (easily) translate into walking turning
                    # print("pupper turning towards last goal")
                    # delta_yaw = self.yaw_from_coords(self.lastGoal[0], self.lastGoal[1], False)
                    # self.rx_ = self.r_alpha * delta_yaw + (1 - self.r_alpha) * self.rx_ #r_alpha*1 for right. r_alpha*-1 for left
                    # command.yaw_rate = self.rx_ * -self.config.max_yaw_rate
                    # super().run(state, command)

                    # v2: back and turn towards old goal
                    print("pupper backing and turning towards last goal")
                    self.ly_ = self.l_alpha * how_far*-1 + (1 - self.l_alpha) * self.ly_         # l_alpha*1 for forward. l_alpha*-1 for backward
                    x_vel = self.ly_ * self.config.max_x_velocity
                    y_vel = self.lx_ * -self.config.max_y_velocity
                    
                    command.horizontal_velocity = np.array([x_vel, y_vel])
                    if state.behavior_state != BehaviorState.WALK:
                        command.walk_event = True
                    delta_yaw = self.yaw_from_coords(self.lastGoal[0], self.lastGoal[1], False)
                    self.rx_ = self.r_alpha * delta_yaw + (1 - self.r_alpha) * self.rx_ #r_alpha*1 for right. r_alpha*-1 for left
                    command.yaw_rate = self.rx_ * -self.config.max_yaw_rate
                    super().run(state, command)
                else: # goal is found & in the middle
                    print ("goal is in the middle, start walking")
                    self.reorient_state = False
                    self.ly_ = 0
                    


            if goal is not None and not self.reorient_state:
                delta_yaw = self.yaw_from_coords(goal[0], goal[1])
                how_far = self.depth_fn(goal[1])                
                
                if how_far != 0:  
                    self.ly_ = self.l_alpha * how_far + (1 - self.l_alpha) * self.ly_         # l_alpha*1 for forward. l_alpha*-1 for backward
                    x_vel = self.ly_ * self.config.max_x_velocity
                    y_vel = self.lx_ * -self.config.max_y_velocity
                    
                    command.horizontal_velocity = np.array([x_vel, y_vel])
        
                    self.rx_ = self.r_alpha * delta_yaw + (1 - self.r_alpha) * self.rx_ #r_alpha*1 for right. r_alpha*-1 for left
                        
                    command.yaw_rate = self.rx_ * -self.config.max_yaw_rate

                    if state.behavior_state != BehaviorState.WALK:
                        command.walk_event = True
                    super().run(state, command)
                else:
                    if state.behavior_state != BehaviorState.REST:
                        print("goal reached")
                        command.stand_event = True
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
