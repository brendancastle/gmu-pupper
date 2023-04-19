#!/usr/bin/env python
import queue
import sys
import threading
import rospy
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.models import detection
import numpy as np
import cv2
import os
import message_filters
from pupper_object_detection.msg import Detections, Detection, Test
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class PupperObjectDetector:

    def __init__(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        

        # jit model to take it from ~20fps to ~30fps
        # net = torch.jit.script(net)
        # Configure depth and color streams
        # pipeline = rs.pipeline()
        # config = rs.config()

        # Get device product line for setting a supporting resolution
        # pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        # # pipeline_profile = config.resolve(pipeline_wrapper)
        # device = pipeline_profile.get_device()
        # device_product_line = str(device.get_info(rs.camera_info.product_line))

        # found_rgb = False
        # for s in device.sensors:
        #     if s.get_info(rs.camera_info.name) == 'RGB Camera':
        #         found_rgb = True
        #         break      
        # if not found_rgb:
        #     rospy.loginfo("The demo requires Depth camera with Color sensor")
        #     exit(0)

        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
        # config.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 6)
        
        model.conf = 0.75  # NMS confidence threshold
        model.iou = 0.45  # NMS IoU threshold
        model.agnostic = False  # NMS class-agnostic
        model.multi_label = False  # NMS multiple labels per box
        # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        model.classes = None
        model.max_det = 1000  # maximum number of detections per image
        model.amp = False  # Automatic Mixed Precision (AMP) inference

        self.model = model
        # self.pipeline = pipeline
        # self.config = config

        self.detectionsPublisher = rospy.Publisher("pupper_detections", Detections)
        # rgbSubscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.handleRgbFrame, queue_size=5)
        # alignedDepthSubscriber = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.handleAlignedDepthFrame, queue_size=5)

        rgbSubscriber = message_filters.Subscriber("/camera/color/image_raw", Image)
        alignedDepthSubscriber = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)

        # #frames are 30fps so look for things within 2/30 seconds of eachother?
        self.synchronizer = message_filters.TimeSynchronizer([rgbSubscriber,alignedDepthSubscriber], 10, 2/30.0)
        self.queue = queue.LifoQueue(maxsize=60) #keep 60 entries at most, 2 seconds of data.
        self.cv_bridge = CvBridge()
        

    def handleFrames(self, rgbImage:Image, depthImage:Image):
        # rospy.loginfo("***************************")
        # rospy.loginfo("RGB: ")
        # rospy.loginfo(rgbFrame.header)
        # rospy.loginfo("Depth: ")
        # rospy.loginfo(depthFrame.header)        
        
        
        self.queue.put((rgbImage,depthImage))
        rospy.logdebug_throttle(1, f"rgb = {rgbImage.encoding} \t depth = {depthImage.encoding}")
        rospy.logdebug_throttle(1, f"Added frames to queue. Size ~= {self.queue.qsize()}")


    def handleRgbFrame(self, rgbFrame:Image):
        """
        Handle /camera/color/image_raw callbacks. This publishes sensor_msgs/Image of the following format:

        std_msgs/Header header
        uint32 seq
        time stamp
        string frame_id
        uint32 height
        uint32 width
        string encoding
        uint8 is_bigendian
        uint32 step
        uint8[] data
        """
        
        rospy.loginfo(f"RGB: {rgbFrame.header}")
        return

    def handleAlignedDepthFrame(self, alignedDepthFrame):
        """
        Handle /camera/aligned_depth_to_color/image_raw callbacks. This publishes sensor_msgs/Image of the following format:

        std_msgs/Header header
        uint32 seq
        time stamp
        string frame_id
        uint32 height
        uint32 width
        string encoding
        uint8 is_bigendian
        uint32 step
        uint8[] data
        """
        
        rospy.loginfo(f"Depth: {alignedDepthFrame.header}")
        return

    def runInference(self, color_image:Image, depth_image:Image):        
        start = time.time()            
        # Convert images to numpy arrays
        depth_frame = self.cv_bridge.imgmsg_to_cv2(depth_image, depth_image.encoding)
        color_frame = self.cv_bridge.imgmsg_to_cv2(color_image, color_image.encoding)
        
        depth_image = np.asanyarray(depth_frame)
        color_image = np.asanyarray(color_frame)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        color_image = color_image[:, :, [2, 1, 0]]
        # input_tensor = preprocess(color_image)

        threshold = 0.75

        # img = preprocess(color_image)

        rospy.loginfo("Passing frame through model...")
        results = self.model([color_image], size=640)
        rospy.loginfo("Done with inference")

        # results.print()
        # results.save()  # or .show()
        detections = []
        
        
        image = color_image.copy()

        for box in results.xyxy[0]:
            if box[4] < threshold:
                continue

            # if box[4]>=90:
            # rospy.loginfo(f"Drawing box for {box[5]}")
            xB = int(box[2])
            xA = int(box[0])
            yB = int(box[3])
            yA = int(box[1])
            bgr_color = (0, 0, 0)
            start_point = [xA, yA]
            end_point = [xB, yB]
            centerPoint = (xB - xA, yB-yA)
            depthAtCenter = depth_frame.get_distance(centerPoint[0], centerPoint[1])
            if self.displayImages:
                image = cv2.rectangle(image, start_point, end_point, bgr_color, 3)
                className = self.model.names[int(box[5])]
                image = cv2.putText(image, f"{className} {centerPoint}", start_point, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv2.LINE_AA)

            detection = Detection()
            detection.xmin = xA
            detection.ymin = yA
            detection.xmax = xB
            detection.ymax = yB
            detection.depthAtCenter = depthAtCenter
            detection.className = className
            detection.classNumber = int(box[5])
            detections.append(detection)
        
        detectionsMsg = Detections()
        detectionsMsg.detections = detections
        self.detectionsPublisher.publish(detectionsMsg)
        # if totalFrames > 0 and totalRunTime > 0:
        #     averageFps = totalFrames / totalRunTime
        #     image = cv2.putText(image, str(averageFps), [
        #                         10, 20], cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # rospy.loginfo(results.pandas().xyxy[0])  # im1 predictions (pandas)

        # if self.displayImages:
        #     if depth_colormap_dim != color_colormap_dim:
        #         resized_color_image = cv2.resize(image, dsize=(
        #             depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        #         images = np.hstack((resized_color_image, depth_colormap))
        #     else:
        #         images = np.hstack((image, depth_colormap))

        # if self.displayImages:
        #     # Show images
        #     cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        #     cv2.imshow('RealSense', image)
        #     cv2.waitKey(1)

        
        # rospy.loginfo(f"Frame time: {end-start}")

    def detectionWorker(self):
        totalFrames = 0
        totalRunTime = 0

        while not self.stopSignal.is_set():            
            (rgbFrame,depthFrame) = self.queue.get()
            start = time.time()
            rospy.loginfo("Got frames...")
            self.runInference(rgbFrame,depthFrame)
            end = time.time()        
            lastFrameTime = end-start
            totalRunTime += lastFrameTime
            totalFrames += 1
            if totalFrames > 0 and totalRunTime > 0:
                averageFps = totalFrames / totalRunTime
                rospy.loginfo(f"FPS: {averageFps}")

    def start(self):
        self.displayImages = False
        self.synchronizer.registerCallback(self.handleFrames)        
        self.lock = threading.Lock()
        self.stopSignal = threading.Event()
        self.detectionWorkerThread = threading.Thread(target=self.detectionWorker)
        self.detectionWorkerThread.start()
                
        rospy.spin()
        self.stopSignal.set()
        self.detectionWorkerThread.join()
            
    
    def stop(self):
        rospy.loginfo(f"Stopping pupper object detector")
        # self.pipeline.stop()


if __name__ == '__main__':
    rospy.init_node("PupperObjectDetector")
    detector = PupperObjectDetector()
    try:
        detector.start()
    except rospy.ROSInterruptException:
        detector.stop()
