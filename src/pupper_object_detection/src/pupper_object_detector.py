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
from pupper_object_detection.msg import Detections, Detection
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from pupper_rgbd_publisher.msg import RgbdBundle
import pathlib


def draw_text(img, text,
        font=cv2.FONT_HERSHEY_PLAIN,
        pos=(0, 0),
        font_scale=3,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0)
        ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

class PupperObjectDetector:

    def __init__(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True) # <- pretrained base model
        modelPath = rospy.get_param("/pupper_object_detector_node/modelPath")
        # model = torch.hub.load('ultralytics/yolov5', 'custom', modelPath) # <- finetuned model
        rospy.loginfo("Model names: ")
        rospy.loginfo(model.names)
        # model.conf = 0.50  # NMS confidence threshold
        # model.iou = 0.45  # NMS IoU threshold
        for i in range(len(model.names)):
            rospy.loginfo(f"{i}\t{model.names[i]}")
        model.agnostic = False  # NMS class-agnostic
        model.multi_label = False  # NMS multiple labels per box
        # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        # 32 = sports ball
        model.classes = [32]
        model.max_det = 5  # maximum number of detections per image
        model.amp = True  # Automatic Mixed Precision (AMP) inference

        self.model = model
        
        self.rbgStart = 0
        
        
        self.detectionsPublisher = rospy.Publisher("pupper_detections", Detections, queue_size=1)
        
        rgbSubscriber = message_filters.Subscriber("/camera/color/image_raw", Image)
        alignedDepthSubscriber = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)

        # rospy.Subscriber("/camera/color/image_raw", Image, self.handleRgbFrames)
        # rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.handleDepthFrames)

        rospy.Subscriber("/pupper_rgbd", RgbdBundle, self.handleRgbdBundle, queue_size=1)

        # #frames are 30fps so look for things within 2/30 seconds of eachother?
        # In practice it seems these tend to have identical timestamps
        self.synchronizer = message_filters.ApproximateTimeSynchronizer([rgbSubscriber,alignedDepthSubscriber], 30, 1)
        
        self.lastFrames = [None, None, None]
        self.lastDetectionMsg = None

        self.counter = 0
        self.currentSeconds = 0
        self.totalInferenceFrames = 0 
        self.totalRunTime = 0
        self.sequence = 0
        self.frameCount = 0
        self.cv_bridge = CvBridge()
        self.frameLock = threading.Lock()
        self.startTime = time.time()
        self.lastSeq = None        
        self.totalInferenceFrames = 0
        self.totalRunTime = 0
        self.averageInferenceFps = 0

    def handleRgbdBundle(self, rgbdBundle):
        rgbdImage = rgbdBundle.rgbImage
        depthImage = rgbdBundle.depthImage
        self.handleRgbFrames(rgbdImage)
        self.handleDepthFrames(depthImage)

    def handleRgbFrames(self, rgbImage):
        if self.rbgStart == 0:
            self.rbgStart = time.time()
        color_frame = self.cv_bridge.imgmsg_to_cv2(rgbImage, rgbImage.encoding)
        color_frame = cv2.resize(color_frame, (640,480))
        color_image = np.asanyarray(color_frame)        
        self.frameCount += 1        
        ellapsed = time.time() - self.rbgStart
        # rospy.loginfo_throttle(1, f"Got RGB: {rgbImage.width}x{rgbImage.height} frames [{rgbImage.header.seq} at {time.time()}")
        rospy.loginfo_throttle(1, f"Got {self.frameCount} rgb frames in {ellapsed} seconds. FPS: {self.frameCount/ellapsed}")
        self.lastFrames[1] = color_image
        self.lastFrames[0] = rgbImage.header.seq

    def handleDepthFrames(self, depthImage):
        depth_frame = self.cv_bridge.imgmsg_to_cv2(depthImage, desired_encoding="passthrough")
        depth_frame = cv2.resize(depth_frame, (640,480))
        depth_image = np.asanyarray(depth_frame)
        self.lastFrames[2] = depth_image
        rospy.loginfo_throttle(1, f"Got D: {depthImage.width}x{depthImage.height} frames [{depthImage.header.seq} at {time.time()}")

    def handleFrames(self, rgbImage:Image, depthImage:Image):
        """
        This is the synchronized callback to receive RGBd frames from the Realsense camera

        Every consumer of these frames downstreams has to convert to a np.ndarray so we just do that here and 
        make them available to everyone
        """       
        
        # rospy.loginfo(rgbImage)
        # rospy.loginfo(depthImage)

        depth_frame = self.cv_bridge.imgmsg_to_cv2(depthImage, desired_encoding="passthrough")
        color_frame = self.cv_bridge.imgmsg_to_cv2(rgbImage, rgbImage.encoding)
        
        depth_image = np.asanyarray(depth_frame)
        color_image = np.asanyarray(color_frame)        
        
        rospy.loginfo_throttle(1, f"Got {rgbImage.width}x{rgbImage.height} frames [{rgbImage.header.seq},{depthImage.header.seq} at {time.time()}")
        self.lastFrames = [rgbImage.header.seq, color_image, depth_image]       
    
    def renderFrames(self):        
        while not self.stopSignal.is_set():
            if self.lastFrames is None or self.lastFrames[0] is None or self.lastFrames[1] is None or self.lastFrames[2] is None:
                rospy.loginfo_throttle(1, "Waiting for rgb frames...")
                continue
            
            seq, color_image, depth_image = self.lastFrames            
            if self.lastSeq == seq:
                rospy.loginfo("Got stale image")
                continue                      
            
            
            image = color_image.copy()
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_depth = depth_image.copy()

            if self.lastDetectionMsg is not None:                
                for detection in self.lastDetectionMsg.detections:
                    xA = detection.xmin 
                    yA = detection.ymin 
                    xB = detection.xmax 
                    yB = detection.ymax 
                    depthAtCenter = detection.depthAtCenter 
                    className = detection.className 

                    bgr_color = (0, 0, 0)
                    start_point = [xA, yA]
                    end_point = [xB, yB]
                
                    image = cv2.rectangle(image, start_point, end_point, bgr_color, 3)                    
                    image = cv2.putText(image, f"{className} {depthAtCenter/1000.0:.2f}m", start_point, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv2.LINE_AA)

            if self.frameCount > 0:
                averageFps = self.frameCount / (time.time() - self.startTime)                
                msg = f"Video FPS: {averageFps:.1f}, Inference FPS: {self.averageInferenceFps:.1f}"                
                draw_text(image, msg, font_scale=1, pos=(10,10))
                # font = cv2.FONT_HERSHEY_PLAIN                
                # textSize, _ = cv2.getTextSize(msg, font, 1, cv2.LINE_AA)
                # rospy.loginfo(f"textSize = {textSize}. Position = {position}")
                # x,y=position
                # textWidth, textHeight = textSize
                # rospy.loginfo(f"x,y = {x},{y} textWidth,textHeight = {textWidth}, {textHeight}")
                # cv2.rectangle(image, position, (x + textWidth, y + textHeight), (0,0,0), -1)
                # image = cv2.putText(image, msg, position, font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image_depth), cv2.COLORMAP_JET)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_CUBIC)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((image, depth_colormap))        
            
            # Show images            
            cv2.namedWindow('PupperVision', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('PupperVision', images)
            cv2.waitKey(1)            

    def runInference(self, color_image:Image, depth_image:Image):        
                
        with torch.no_grad():
            results = self.model([color_image], size=640)

        detections = []
        
        for box in results.xyxy[0]:
            confidence = float(box[4])

            xB = int(box[2])
            xA = int(box[0])
            yB = int(box[3])
            yA = int(box[1])
            
            centerPoint = (int(yA+(yB-yA)/2),int(xA+(xB - xA)/2))
            depthAtCenter = depth_image.item(centerPoint)
            className = self.model.names[int(box[5])]
            # rospy.loginfo_throttle(1, f"Detected a {int(box[5])}: {className}, Centerpoint: {centerPoint} Depth: {depthAtCenter}mm / {depthAtCenter*0.0393701}\" Confidence: {confidence}")

            detection = Detection()
            detection.confidence = confidence
            detection.xmin = xA
            detection.ymin = yA
            detection.xmax = xB
            detection.ymax = yB
            detection.depthAtCenter = depthAtCenter
            detection.className = className
            detection.classNumber = int(box[5])
            detections.append(detection)            
        
        detectionsMsg = Detections()
        
        #This just sorts detections in order of decreasing confidence
        detectionsMsg.detections = list(reversed(sorted(detections, key = lambda d: d.confidence)))
        self.lastDetectionMsg = detectionsMsg
        self.detectionsPublisher.publish(detectionsMsg)

    def detectionWorker(self):
        self.totalInferenceFrames = 0
        self.totalRunTime = 0
        self.averageInferenceFps = 0
        while not self.stopSignal.is_set():                        
            if self.lastFrames == None or self.lastFrames[0] is None or self.lastFrames[1] is None or self.lastFrames[2] is None:
                rospy.loginfo_throttle(1, "Waiting for frames...")
                continue
            
            seq, rgbFrame, depthFrame = self.lastFrames
            
            start = time.time()            
            
            self.runInference(rgbFrame,depthFrame)            
            
            end = time.time()        
            lastFrameTime = end-start
            self.totalRunTime += lastFrameTime
            self.totalInferenceFrames += 1
            if self.totalInferenceFrames > 0 and self.totalRunTime > 0:
                self.averageInferenceFps = self.totalInferenceFrames / self.totalRunTime
                rospy.loginfo_throttle(1, f"FPS: {self.averageInferenceFps}")

    def start(self):
        self.displayImages = rospy.get_param("pupper_object_detector_node/displayImages",False)
        # self.synchronizer.registerCallback(self.handleFrames)        
        self.lock = threading.Lock()
        self.stopSignal = threading.Event()
        self.detectionWorkerThread = threading.Thread(target=self.detectionWorker)
        self.detectionWorkerThread.start()

        self.renderingThread = threading.Thread(target=self.renderFrames)
        self.renderingThread.start()

                
        rospy.spin()
        self.stopSignal.set()
        self.detectionWorkerThread.join()
        self.renderingThread.join()
            
    
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


