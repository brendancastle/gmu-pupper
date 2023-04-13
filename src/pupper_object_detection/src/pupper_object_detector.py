#!/usr/bin/env python
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
from pupper_object_detection.msg import Detection, Detections, Test


class PupperObjectDetector:

    def __init__(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        # jit model to take it from ~20fps to ~30fps
        # net = torch.jit.script(net)
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
            rospy.loginfo("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
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
        self.pipeline = pipeline
        self.config = config

        self.detectionsPublisher = rospy.Publisher("pupper_detections_publisher", Detections)
        testPublisher = rospy.Publisher("test_publisher", Test)
        t = Test()
        t.number = 86
        t.label = "Jack"
        testPublisher.publish(t)

    def start(self):
        # Start streaming
        self.pipeline.start(self.config)
        totalRunTime = 0
        totalFrames = 0
        displayImages = True

        while not rospy.is_shutdown():
            start = time.time()
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            # if not depth_frame or not color_frame:
            #     continue
            if not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)
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
                if displayImages:
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
            if totalFrames > 0 and totalRunTime > 0:
                averageFps = totalFrames / totalRunTime
                image = cv2.putText(image, str(averageFps), [
                                    10, 20], cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            # rospy.loginfo(results.pandas().xyxy[0])  # im1 predictions (pandas)

            if displayImages:
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(image, dsize=(
                        depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((image, depth_colormap))

            if displayImages:
                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', image)
                cv2.waitKey(1)

            end = time.time()
            lastFrameTime = end-start
            totalRunTime += lastFrameTime
            totalFrames += 1
            if totalFrames > 0 and totalRunTime > 0:
                averageFps = totalFrames / totalRunTime
                rospy.loginfo(f"FPS: {averageFps}")
            # rospy.loginfo(f"Frame time: {end-start}")
    
    def stop(self):
        rospy.loginfo(f"Stopping pupper object detector")
        self.pipeline.stop()


if __name__ == '__main__':
    rospy.init_node("PupperObjectDetector")
    detector = PupperObjectDetector()
    try:
        detector.start()
    except rospy.ROSInterruptException:
        detector.stop()
