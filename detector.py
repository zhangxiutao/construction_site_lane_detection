#!/usr/bin/env python
from __future__ import print_function

# import roslib
# roslib.load_manifest('my_package')
import os
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from construction_site_lane_detection.msg import Detection
from construction_site_lane_detection.msg import ImageDetections
from darknet_ros_msgs.msg import ImageWithBoundingBoxes
from cv_bridge import CvBridge, CvBridgeError
from lane import *


class LeitbakeDetector:

  def __init__(self):
    
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/darknet_ros/detection_image",ImageWithBoundingBoxes,self.callback,queue_size = 1)
    self.bounding_boxes_pub = rospy.Publisher("bounding_boxes_leitbake",ImageDetections)

  def callback(self,data):

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data.image, "bgr8")
      
    except CvBridgeError as e:
      print(e)

    bounding_boxes = data.bounding_boxes.bounding_boxes

    (rows,cols,channels) = cv_image.shape

    detections = process_frame(cv_image,bounding_boxes)
    imageDetections = ImageDetections()

    for (x1,y1,x2,y2) in detections:
          
      detections_msg = Detection()
      detections_msg.x1 = x1
      detections_msg.y1 = y1
      detections_msg.x2 = x2 
      detections_msg.y2 = y2
      imageDetections.detections.append(detections_msg)

    self.bounding_boxes_pub.publish(imageDetections)

def main(args):
  
  rospy.init_node('LeitbakeDetector', anonymous=True)
  ld = LeitbakeDetector()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)