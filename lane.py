#!/usr/bin/env python
"""Departure Warning System with a Monocular Camera"""

__author__ = "Junsheng Fu"
__email__ = "junsheng.fu@yahoo.com"
__date__ = "March 2017"


import numpy as np
import cv2
import matplotlib.pyplot as plt
import cnn_classifier
from timeit import default_timer as timer
from calibration import load_calibration
from copy import copy

def chekIfOverlaying(bb1,bb2):

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return False
    else:
        return True

leitbake_detector = cnn_classifier.CnnClassifier()

def checkIfTailLight(bounding_box_car,bounding_box_leitbake):
    bb_car = {'x1':bounding_box_car.xmin,'y1':bounding_box_car.ymin,'x2':bounding_box_car.xmax,'y2':bounding_box_car.ymax}
    bb_leitbake = {'x1':bounding_box_leitbake[0],'y1':bounding_box_leitbake[1],'x2':bounding_box_leitbake[2],'y2':bounding_box_leitbake[3]}
    if chekIfOverlaying(bb_car,bb_leitbake):
        return True
    else:
        return False


def process_frame(img_cv2,bounding_boxes):

    start = timer()


    ret = leitbake_detector.leibake_detect(img_cv2,bounding_boxes)
    if ret is not None:
        detections,img_lowerhalf_cv2 = ret
        for (x1,y1,x2,y2) in detections:
            cv2.rectangle(img_lowerhalf_cv2,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow("leitbake_detected",img_lowerhalf_cv2)
        cv2.waitKey(1)
        end = timer()
        #print "fps is {}".format(1/(end-start))
        return detections
    else:
        return None
    



