from logging import debug
from time import sleep
import sys
from pybullet_utils import bullet_client
import pybullet
import json
import msgpack
from serial import EIGHTBITS, PARITY_NONE, STOPBITS_ONE
import math 

PYBULLET_KP_GAIN = 4
PYBULLET_KD_GAIN = 0.2

class Camera_serial:

    def __init__(self, cam_skip_frames, distance, show_camera=False):
        self.cam_skip=0
        self.cam_skip_frames = cam_skip_frames
        self.distance = distance

        self.img_w, self.img_h = 120, 80

        self._timeout = 0
        self._written = None
        self._line_term = '\r\n'
        self._rate = 0.01
        debug(f'Setup get_camera write pusher')
        print("creating pupper drive system")
        
        self.p = bullet_client.BulletClient()
        self.pupper_body_uid = -1
        self.pupper_link_indices=[]
        self.p.setTimeStep(0.001)
        #self.p.setPhysicsEngineParameter(numSubSteps=1)

        for i in range (self.p.getNumBodies()):
          b = self.p.getBodyUniqueId(i)
          print("body ", b, self.p.getBodyInfo(b))
          if self.p.getBodyInfo(b)[1]==b'pupper_v2_dji':
            print("found Pupper in camera module")
            found_pupper_sim = True
            self.pupper_body_uid = b
            
        self.time_stamp = 0
        if not found_pupper_sim:
          sys.exit("Error: Cannot find pupper, is pupper_server running?")
        #self.p.configureDebugVisualizer(rgbBackground=[0,1,0])

        if show_camera:
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 1)

        self.distance_calculated_from_pybullet = True
        

    def get_camera_details(self):

        if self.distance_calculated_from_pybullet:
            max_distance = 10000

            for i in range (self.p.getNumBodies()):
                b = self.p.getBodyUniqueId(i)
                if self.p.getBodyInfo(b)[1]==b'cone':
                    cone_id = b
                    break

            closest_points = self.p.getClosestPoints(self.pupper_body_uid, cone_id, max_distance)

            min_dist = max_distance
            object_pos = None
            for i in range(len(closest_points)):
                if min_dist > closest_points[i][8]:
                    min_dist = closest_points[i][8]
                    object_pos = closest_points[i][6]
            

            pos, orn = self.p.getBasePositionAndOrientation(self.pupper_body_uid)
            yaw_0 = self.p.getEulerFromQuaternion(orn)[2]
            
            theta1 = math.atan2(object_pos[1]-pos[1],object_pos[0]-pos[0])
            def limit_fn(x):
                if x> 2*math.pi or x < -2*math.pi:
                    raise Exception(f"limit_fn not defined for x>2pi or x<-2pi. value of x: {x:.4f}")
                if -math.pi < x and x < math.pi:
                    return x
                elif math.pi < x:
                    return -math.pi + (x%math.pi)
                else:
                    return math.pi + (x%-math.pi)
            def wrap_around_fn(x):
                if x> 2*math.pi or x < -2*math.pi:
                    raise Exception(f"wrap_around_fn not defined for x>2pi or x<-2pi. value of x: {x:.4f}")
                if -math.pi < x and x < math.pi:
                    return x
                elif math.pi < x:
                    return -2*math.pi + x
                else:
                    return 2*math.pi + x

            a_hat, b_hat = 0.99834, 1.5696
            calculated_yaw = theta1*a_hat + b_hat
            calculated_yaw = limit_fn(calculated_yaw)
            delta_yaw = wrap_around_fn(calculated_yaw-yaw_0)
            
            return delta_yaw, min_dist
        else:
            if (self.cam_skip>self.cam_skip_frames):
                agent_pos, agent_orn = self.p.getBasePositionAndOrientation(self.pupper_body_uid)

                yaw = self.p.getEulerFromQuaternion(agent_orn)[0]
                xA, yA, zA = agent_pos
                zA = zA + 0.07
                xA = xA +0.09

                xB = xA + math.cos(yaw)*self.distance
                yB = yA + math.sin(yaw)*self.distance
                zB = zA 
                
                view_matrix = self.p.computeViewMatrix(cameraEyePosition=[xA,yA,zA], cameraTargetPosition=[xB,yB,zB], cameraUpVector=[0,0,1.0])

                projection_matrix = self.p.computeProjectionMatrixFOV(fov=90, aspect=1.5, nearVal=0.02, farVal=3.5)
                imgs = self.p.getCameraImage(self.img_w, self.img_h, view_matrix, projection_matrix, renderer=self.p.ER_BULLET_HARDWARE_OPENGL)
                _,_, rgb, depth, segmentation = imgs
                self.cam_skip = 0

                return rgb, depth
            self.cam_skip += 1
            return None, None