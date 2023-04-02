from src.Controller import Controller
from src.State import BehaviorState
import numpy as np

class FollowController(Controller):
    def __init__(self, config, inverse_kinematics):
        super().__init__(config, inverse_kinematics)

        self.in_follow_state = False
        self.following = False
        self.lx_ = 0
        self.ly_ = 0


        # updating transition mappings with FOLLOW
        self.trot_transition_mapping[BehaviorState.FOLLOW] = BehaviorState.TROT
        self.walk_transition_mapping[BehaviorState.FOLLOW] = BehaviorState.TROT
        self.stand_transition_mapping[BehaviorState.FOLLOW] = BehaviorState.TROT

        self.follow_transition_mapping = {
            BehaviorState.REST: BehaviorState.FOLLOW,
            BehaviorState.TROT: BehaviorState.FOLLOW,
            BehaviorState.WALK: BehaviorState.FOLLOW,
            BehaviorState.FOLLOW: BehaviorState.FOLLOW
        }

        self.within_follow_transition_mapping = {
            BehaviorState.REST: BehaviorState.WALK,
            BehaviorState.WALK: BehaviorState.REST,
        }
    
    def run(self, state, command):
        if command.follow_event:
            if self.in_follow_state == False:
                # rest until start_following_event
                state.behavior_state = self.stand_transition_mapping[state.behavior_state]
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
                state.behavior_state = self.stand_transition_mapping[state.behavior_state]
                command.stand_event = True
                self.following = False
                super().run(state, command)
                print("u pressed, following exited")

            if self.following:
                # call object detection model. see if model returns a depth, direction. 
                # depth is how far the ball is.
                # direction is to what angle
                depth = 5 - self.count*0.001
                # print("depth now:", depth)
                direction = (0,0) # assume to go straight first
                # calculate commands based on depth, direction
                # pass to super
                
                self.eps = 1
                self.slow_down_distance = 2
                
                def calc_how_far(d):
                    if d > self.slow_down_distance:
                        return 1
                    if d < self.eps:
                        return 0
                    return (d-self.eps)/(self.slow_down_distance-self.eps)
                how_far = calc_how_far(depth)

                l_alpha = 0.15
                # lets just move right and straight
                self.lx_ = l_alpha * 1 + (1 - l_alpha) * self.lx_ 

                # self.ly_ = l_alpha * 1 + (1 - l_alpha) * self.ly_ 
                # self.msg["ly"] = self.ly_

                # x_vel = msg["ly"] * self.config.max_x_velocity
                y_vel = self.lx_ * -self.config.max_y_velocity*how_far

                if y_vel != 0:
                    command.horizontal_velocity = np.array([0, y_vel])
                    state.behavior_state = self.walk_transition_mapping[state.behavior_state]
                    super().run(state, command)
                    self.count += 1
                else:
                    print("goal reached!")
                    state.behavior_state = self.stand_transition_mapping[state.behavior_state]
                    command.stand_event = True
                    super().run(state, command)
                    
                
        else:
            super().run(state, command)
