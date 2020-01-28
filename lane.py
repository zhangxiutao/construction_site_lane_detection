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

def get_iou(bb1, bb2):
    print(bb1)
    print(bb2)
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
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
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

leitbake_detector = cnn_classifier.CnnClassifier()

def checkIfTailLight(bounding_box_car,bounding_box_leitbake):
    bb_car = {'x1':bounding_box_car.xmin,'y1':bounding_box_car.ymin,'x2':bounding_box_car.xmax,'y2':bounding_box_car.ymax}
    bb_leitbake = {'x1':bounding_box_leitbake[0],'y1':bounding_box_leitbake[1],'x2':bounding_box_leitbake[2],'y2':bounding_box_leitbake[3]}
    iou = get_iou(bb_car,bb_leitbake)
    print(iou)
    if iou > 0.5:
        return True
    else:
        return False

def process_frame(img_cv2,bounding_boxes):

    start = timer()

    #height, width = img_cv2.shape[0:2]
    #img_cv2 = img_cv2[int(height/2):height,0:width,:]

    detections = leitbake_detector.leibake_detect(img_cv2)

    for bounding_box in bounding_boxes:
        if bounding_box.Class == "car":
            cv2.rectangle(img_cv2,(bounding_box.xmin,bounding_box.ymin),(bounding_box.xmax,bounding_box.ymax),(255,0,0),2)
            for detection in detections:
                if checkIfTailLight(bounding_box,detection):
                    detections.remove(detection)

    for (x1,y1,x2,y2) in detections:
        cv2.rectangle(img_cv2,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.imshow("leitbake_detected",img_cv2)
    cv2.waitKey(1)
    end = timer()
    print "fps is {}".format(1/(end-start))
    return detections
    



