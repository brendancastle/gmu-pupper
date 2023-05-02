#!/usr/bin/env python
import time
import pyrealsense2 as rs

import rospy
import cv2
import cv_bridge
from pupper_rgbd_publisher.msg import RgbdBundle

# Start streaming

framecount = 0
starttime = time.time()
class PupperRgbdPublisher:

    def __init__(self):
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config = config
        self.pipeline = pipeline
        self.rgbdPublisher = rospy.Publisher("pupper_rgbd", RgbdBundle, queue_size=1)
        self.cvBridge = cv_bridge.CvBridge()

    def start(self):
        self.pipeline.start(self.config)
        framecount = 0
        starttime = time.time()
        while not rospy.is_shutdown():
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            color_array = np.asanyarray(color_frame.get_data())
            depth_array = np.asanyarray(depth_frame.get_data())

            rgbImage = self.cvBridge.cv2_to_imgmsg(color_array, 'bgr8')
            depthImage = self.cvBridge.cv2_to_imgmsg(depth_array, 'passthrough')

            if not depth_frame or not color_frame:
                continue

            bundle = RgbdBundle()
            bundle.rgbImage = rgbImage
            bundle.depthImage = depthImage
            self.rgbdPublisher.publish(bundle)
            framecount +=1
            ellapsed=time.time()-starttime
            rospy.loginfo_throttle(1, f"FPS: {framecount/ellapsed}")
    def stop(self):
        rospy.loginfo(f"Stopping pupper object detector")
        # self.pipeline.stop()


if __name__ == '__main__':
    rospy.init_node("PupperRgbdPublisher")
    detector = PupperRgbdPublisher()
    try:
        detector.start()
    except rospy.ROSInterruptException:
        detector.stop()
