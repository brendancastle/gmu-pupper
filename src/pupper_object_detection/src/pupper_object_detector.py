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

import pathlib

class PupperObjectDetector:

    def __init__(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True) # <- pretrained base model
        modelPath = rospy.get_param("/pupper_object_detector_node/modelPath")
        # model = torch.hub.load('ultralytics/yolov5', 'custom', modelPath) # <- finetuned model
        rospy.loginfo("Model names: ")
        rospy.loginfo(model.names)
        # model.conf = 0.50  # NMS confidence threshold
        # model.iou = 0.45  # NMS IoU threshold
        model.agnostic = False  # NMS class-agnostic
        model.multi_label = False  # NMS multiple labels per box
        # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        # 32 = sports ball
        model.classes = [32]
        model.max_det = 5  # maximum number of detections per image
        model.amp = True  # Automatic Mixed Precision (AMP) inference

        self.model = model
        
        self.detectionsPublisher = rospy.Publisher("pupper_detections", Detections)
        rgbSubscriber = message_filters.Subscriber("/camera/color/image_raw", Image)
        alignedDepthSubscriber = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)

        # #frames are 30fps so look for things within 2/30 seconds of eachother?
        # In practice it seems these tend to have identical timestamps
        self.synchronizer = message_filters.TimeSynchronizer([rgbSubscriber,alignedDepthSubscriber], 10, 2/30.0)
        #Buffering two seconds of data if we want to do anything with that.
        # In practice, we just grab the latest frame. 
        self.queue = queue.LifoQueue(maxsize=60) #keep 60 entries at most, 2 seconds of data.
        self.cv_bridge = CvBridge()
        

    def handleFrames(self, rgbImage:Image, depthImage:Image):
        # rospy.loginfo("***************************")
        # rospy.loginfo("RGB: ")
        # rospy.loginfo(rgbFrame.header)
        # rospy.loginfo("Depth: ")
        # rospy.loginfo(depthFrame.header)        
        
        
        self.queue.put((rgbImage,depthImage))
        rospy.loginfo_throttle(1, f"rgb = {rgbImage.encoding} \t depth = {depthImage.encoding}")
        rospy.loginfo_throttle(1, f"Added frames to queue. Size ~= {self.queue.qsize()}")


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
        
        # rospy.loginfo(f"RGB: {rgbFrame.header}")
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
        
        # rospy.loginfo(f"Depth: {alignedDepthFrame.header}")
        return

    def runInference(self, color_image:Image, depth_image:Image):        
        # Convert images to numpy arrays
        depth_frame = self.cv_bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")
        color_frame = self.cv_bridge.imgmsg_to_cv2(color_image, color_image.encoding)

        
        depth_image = np.asanyarray(depth_frame)
        color_image = np.asanyarray(color_frame)
        # depth_image = (480, 640), color_image = (480, 640, 3)

        # rospy.loginfo(f"depth_image = {depth_image.shape}, color_image = {color_image.shape}")

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    

        # color_image = color_image[:, :, [2, 1, 0]]        

        rospy.logdebug("Passing frame through model...")
        results = self.model([color_image], size=320)
        rospy.logdebug("Done with inference")

        detections = []
        
        image = color_image.copy()
        

        for box in results.xyxy[0]:
            confidence = box[4]

            xB = int(box[2])
            xA = int(box[0])
            yB = int(box[3])
            yA = int(box[1])
            bgr_color = (0, 0, 0)
            start_point = [xA, yA]
            end_point = [xB, yB]
            centerPoint = (int(yA+(yB-yA)/2),int(xA+(xB - xA)/2))
            # rospy.loginfo(depth_image.shape)            
            depthAtCenter = depth_image.item(centerPoint)
            # rospy.loginfo(depthAtCenter)
            className = self.model.names[int(box[5])]
            rospy.loginfo(f"Detected a {int(box[5])}: {className}, Centerpoint: {centerPoint} Depth: {depthAtCenter}mm / {depthAtCenter*0.0393701}\" Confidence: {confidence}")
            if self.displayImages:
                image = cv2.rectangle(image, start_point, end_point, bgr_color, 3)
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
        rospy.loginfo(f"Detected {len(detections)} objects")
        detectionsMsg = Detections()
        detectionsMsg.detections = detections
        self.detectionsPublisher.publish(detectionsMsg)
        if self.totalFrames > 0 and self.totalRunTime > 0:
            averageFps = self.totalFrames / self.totalRunTime
            image = cv2.putText(image, str(averageFps), [10, 20], cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # rospy.loginfo(results.pandas().xyxy[0])  # im1 predictions (pandas)

        if self.displayImages:
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image), cv2.COLORMAP_COOL)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((image, depth_colormap))        
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

    def detectionWorker(self):
        self.totalFrames = 0
        self.totalRunTime = 0

        while not self.stopSignal.is_set():            
            (rgbFrame,depthFrame) = self.queue.get()
            start = time.time()
            # rospy.loginfo("Got frames...")
            self.runInference(rgbFrame,depthFrame)
            end = time.time()        
            lastFrameTime = end-start
            self.totalRunTime += lastFrameTime
            self.totalFrames += 1
            if self.totalFrames > 0 and self.totalRunTime > 0:
                averageFps = self.totalFrames / self.totalRunTime
                rospy.loginfo(f"FPS: {averageFps}")

    def start(self):
        self.displayImages = rospy.get_param("pupper_object_detector_node/displayImages",False)
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
