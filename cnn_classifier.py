# Import the required modules
from __future__ import division
import cv2
import argparse as ap
from config import *
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.ops as ops
import torch.nn
import os.path
import cnn
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import uuid
import time
import random
import math
import imutils
window_area = min_wdw_sz[0]*min_wdw_sz[1]
def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def increase_brightness(img, value=30):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def get_contours_center(cnts):

    centers = []
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            # cv2.drawContours(backtorgb, [c], -1, (0, 255, 0), 2)
            # cv2.circle(backtorgb, (cX, cY), 7, (0, 0, 255), -1)
            centers.append((cX,cY))

    return centers

def red_mask(img_cv2):
    
    img_hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
    #red mask0
    lower_red = np.array([0, 43, 46]) #0, 43, 46
    upper_red = np.array([10, 255, 255]) #10, 255, 255
    mask0 = cv2.inRange(img_hsv,lower_red,upper_red)
    #red mask1
    lower_red = np.array([156, 43, 46]) #156, 43, 46
    upper_red = np.array([180, 255, 255]) #180, 255, 255
    mask1 = cv2.inRange(img_hsv,lower_red,upper_red)

    mask_red = mask0 + mask1
    res_red = cv2.bitwise_and(img_cv2, img_cv2, mask=mask_red)
     
    #kernel = np.ones((10,10),np.uint8)
    # res_red = cv2.erode(res_red,kernel)
    # res_red = cv2.morphologyEx(res_red, cv2.MORPH_CLOSE, kernel)
    h,s,red_gray=cv2.split(res_red)

    return red_gray

def cv22PIL(img_cv2):

    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv2)

    return img_pil

def PIL2cv2(img_pil):

    img_cv2 = np.array(img_pil.convert('RGB'))[:, :, ::-1].copy() 

    return img_cv2

def drawLines(img_cv2,lines):

    for line in lines: 
        rho, theta = line[0]
        angle = theta*180/np.pi
        if angle > 80 and angle < 100:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_cv2, (x1, y1), (x2, y2), (0, 0, 255))

    return img_cv2

def checkIfTooRed(img_window_pil):
    img_window_cv2 = PIL2cv2(img_window_pil)
    arr_std = np.std(img_window_cv2,ddof=1)
    # if arr_std < 55:
    #     return True
    # else:
    return False

def windows_search_contourbased(img_binary_cv2, window_size):
    
    pad_size = 30 
    height, width = img_binary_cv2.shape[0:2]
    cnts = cv2.findContours(img_binary_cv2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    detections = []


    scores = []
    contour_centers = get_contours_center(cnts)
    count = 0
    for (cX,cY) in contour_centers:
        img_window_mask = np.zeros_like(img_binary_cv2)
        count = count + 1
        p1 = (int(cX-math.floor(window_size[0]/2)),int(cY-math.floor(window_size[1]/2)))
        p2 = (int(cX+math.ceil(window_size[0]/2)),int(cY+math.ceil(window_size[1]/2)))
        cv2.rectangle(img_window_mask, p1, p2, 255, thickness=cv2.FILLED)
        img_window_masked = cv2.bitwise_and(img_binary_cv2, img_binary_cv2, mask=img_window_mask)
        cnts_window_masked = cv2.findContours(img_window_masked.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts_window_masked = imutils.grab_contours(cnts_window_masked)
        contour_centers_window_masked = get_contours_center(cnts_window_masked)
        x1_corrected = p1[0]
        y1_corrected = p1[1]
        x2_corrected = p2[0]
        y2_corrected = p2[1]
        if len(contour_centers_window_masked) >= 2:

            if p1[0] >= -pad_size and p1[1] >= -pad_size and p2[0] <= (width+pad_size) and p2[1] <= (height+pad_size):

                # if p1[0] < 0 and p1[0] >= -pad_size and p1[1] < 0 and p1[1] >= -pad_size:

                #     x1_corrected = 0
                #     y1_corrected = 0
                #     x2_corrected = x1_corrected+window_size[0]
                #     y2_corrected = y1_corrected+window_size[1]

                # if p2[0] < width and p1[0] >= 0 and p1[1] < 0 and p1[1] >= -pad_size:
                    
                #     x1_corrected = p1[0]
                #     y1_corrected = p1[1]
                #     x2_corrected = p2[0]
                #     y2_corrected = p2[1]   

                # if p1[0] < 0 and p1[0] >= -pad_size and p2[1] >= width and p2[1] <= (width+pad_size):
                    
                #     x1_corrected = width-1-window_size[0]
                #     y1_corrected = 0
                #     x2_corrected = width-1
                #     y2_corrected = window_size[1]

                # if  p2[0] <= (width+pad_size) and p2[1] <= height and p1[1] >= 0:

                #     x1_corrected = p1[0]
                #     y1_corrected = p1[1]
                #     x2_corrected = width-1
                #     y2_corrected = p2[1]                       

                # if  p2[0] <= (width+pad_size) and p2[0] >= width and p2[1] <= (height+pad_size) and p1[1] >= height:

                #     x1_corrected = width-1-window_size[0]
                #     y1_corrected = height-1-window_size[1]
                #     x2_corrected = width-1
                #     y2_corrected = height-1                      

                # if  p2[0] <= (width+pad_size) and p2[0] >= width and p2[1] <= (height+pad_size) and p1[1] >= height:

                #     x1_corrected = width-1-window_size[0]
                #     y1_corrected = height-1-window_size[1]
                #     x2_corrected = width-1
                #     y2_corrected = height-1                   

                if  p2[1] <= (height+pad_size) and p2[1] >= height:

                    y1_corrected = height-1-window_size[1]
                    y2_corrected = height-1                           
                    
                if  p1[1] < 0 and p1[1] >= -pad_size:

                    y1_corrected = 0
                    y2_corrected = window_size[1]                         
                                    
                if p1[0] < 0 and p1[0] >= -pad_size:
                    
                    x1_corrected = 0
                    x2_corrected = window_size[0]

                if p2[0] >= width and p2[0] <= (width+pad_size):
                    
                    x1_corrected = width-1-window_size[1]
                    x2_corrected = width-1
                
                detections.append((x1_corrected,y1_corrected,x2_corrected,y2_corrected))
                y_mean_centers = np.mean(contour_centers_window_masked,axis=0)[1].astype(np.int32)
                if (y_mean_centers-cY) != 0:
                    score = 1/((y_mean_centers-cY)**2)
                    scores.append(score)
                else:
                    scores.append(0)

    detections = torch.tensor(detections).double()
    scores = torch.tensor(scores).double()
    detections_nms_idx = ops.nms(detections,scores,0.2)
    detections_nms = [detections[i].tolist() for i in detections_nms_idx]
    detections_nms = [[int(x) for x in detection_nms] for detection_nms in detections_nms]

    for (x1,y1,x2,y2) in detections_nms:
        cv2.rectangle(img_binary_cv2, (x1,y1), (x2,y2), 255, thickness=1)

    return detections_nms

def draw_hough_lines(img_binary_cv2):

    hough_lines = cv2.HoughLinesP(img_binary_cv2, 1, np.pi / 180, threshold = 1,minLineLength = 50,maxLineGap = 200)
    hough_lines = np.squeeze(hough_lines)
    hough_lines_filtered = []
    if hough_lines[()] is not None:
        if hough_lines.ndim == 1:
            hough_lines = hough_lines.reshape(1,4)
        for x1,y1,x2,y2 in hough_lines:
            if x1 != x2:
                angle_line = abs(math.atan((y1-y2)/(x2-x1))*180/math.pi)
            else:
                angle_line = 90
            if angle_line < 18:
                hough_lines_filtered.append((x1,y1,x2,y2))

    return hough_lines_filtered     

def random_window(img_origin_pil, img_cv2, window_size):

    count = 0
    nonzeros = cv2.findNonZero(img_cv2)
    if nonzeros is None:
        return None
    windows_pil = []
    found = False
    for count in range(10):
        anchor_point = random.choice(nonzeros)[0]
        mid_p1 = (anchor_point[0]-math.floor(window_size[0]/2),anchor_point[1]-math.floor(window_size[1]/2))
        mid_p2 = (anchor_point[0]+math.ceil(window_size[0]/2),anchor_point[1]+math.ceil(window_size[1]/2))
        if mid_p1[0] > 0 and mid_p1[1] > 0 and mid_p2[0] < img_origin_pil.size[0] and mid_p2[1] < img_origin_pil.size[1]:
            found = True    
            break
    if not found:
        return None
    if mid_p1[1]+int(window_size[1]/2) < img_origin_pil.size[1]:
        lower_window_pil = img_origin_pil.crop((mid_p1[0],mid_p1[1]+int(window_size[1]/2),mid_p2[0],mid_p2[1]+int(window_size[1]/2)))
        if checkIfTooRed(lower_window_pil):
            lower_window_pil = None
    else:
        lower_window_pil = None

    mid_window_pil = img_origin_pil.crop((mid_p1[0],mid_p1[1],mid_p2[0],mid_p2[1]))
    if checkIfTooRed(mid_window_pil):
        mid_window_pil = None
    if mid_p1[1]-int(window_size[1]/2) > 0:
        upper_window_pil = img_origin_pil.crop((mid_p1[0],mid_p1[1]-int(window_size[1]/2),mid_p2[0],mid_p2[1]-int(window_size[1]/2)))
        if checkIfTooRed(upper_window_pil):
            upper_window_pil = None
    else:
        upper_window_pil = None

    windows_pil.append(lower_window_pil)
    windows_pil.append(mid_window_pil)
    windows_pil.append(upper_window_pil)

    return (mid_p1[0],mid_p1[1],windows_pil)

class CnnClassifier:

    def __init__(self):

        self.test_on_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.test_on_gpu else "cpu")
        self.model = cnn.Net()
        self.model.to(self.device)
        print(self.device)
        self.model.double()
        self.model.load_state_dict(torch.load("./src/construction_site_lane_detection/models/nn_model.pt"))
        self.model.eval()
        self.state = 1 #there is a red truck in front
        self.frame_count = 0

    def leibake_detect(self,img_origin_cv2,bounding_boxes):
        # List to store the detections
        detections = []
        detections_nms = []
        scores = []
        imgs_candidate_window_pil = []
        toTensor = transforms.ToTensor()
        norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # This list contains detections at the current scale
        cd = []

        height, width = img_origin_cv2.shape[0:2]
        lowerhalf_height = int(height/4)
        lowerhalf_width = width

        if (self.frame_count % 100) == 0:
            self.frame_count = 1
            self.img_car_mask = np.ones([lowerhalf_height,lowerhalf_width]).astype(np.uint8)
        else:
            self.frame_count = self.frame_count+1

        img_lowerhalf_cv2 = img_origin_cv2[height-lowerhalf_height:(height),0:lowerhalf_width,:]
        img_lowerhalf_pil = cv22PIL(img_lowerhalf_cv2)

        img_lowerhalf_red_masked_cv2 = red_mask(img_lowerhalf_cv2)       
        for bounding_box in bounding_boxes:
            if bounding_box.Class == "car" or bounding_box.Class == "truck" or bounding_box.Class == "train" or bounding_box.Class == "stop sign":
                bounding_box.ymin = bounding_box.ymin-int(3*height/4)
                bounding_box.ymax = bounding_box.ymax-int(3*height/4)
                
                self.img_car_mask[max(0,bounding_box.ymin):max(0,bounding_box.ymax),bounding_box.xmin:bounding_box.xmax] = 0
        self.img_car_mask[self.img_car_mask==1] = 255

        if (self.frame_count % 10) == 0:

            img_lowerhalf_car_masked_cv2 = cv2.bitwise_and(img_lowerhalf_red_masked_cv2,img_lowerhalf_red_masked_cv2,mask=self.img_car_mask)

            ret,img_thresh_cv2=cv2.threshold(img_lowerhalf_car_masked_cv2,0,255,0)

            cnts_before_cntfiltered, hierarchy = cv2.findContours(img_thresh_cv2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            img_contourfiltered = img_thresh_cv2

            for contour in cnts_before_cntfiltered:
                if cv2.contourArea(contour)>150 or cv2.contourArea(contour)<20:
                    img_contourfiltered = cv2.drawContours(img_contourfiltered, [contour], -1, 0, thickness=cv2.FILLED)
     
            coordinates_windows = windows_search_contourbased(img_contourfiltered,min_wdw_sz)
            cv2.imshow("contourfiltered",img_contourfiltered)
            print(coordinates_windows)
            if len(coordinates_windows)>=3:

                print("construction site candidate frame found!"+str(self.frame_count))

                for coordinate_window in coordinates_windows:
                    #print([coordinate_window[1]+height-lowerhalf_height,coordinate_window[3]+height-lowerhalf_height, coordinate_window[0],coordinate_window[2]])
                    print(coordinate_window[1]+height-lowerhalf_height,coordinate_window[3]+height-lowerhalf_height)
                    print(height)
                    img_window_pil = cv22PIL(img_origin_cv2[coordinate_window[1]+height-lowerhalf_height:coordinate_window[3]+height-lowerhalf_height, coordinate_window[0]:coordinate_window[2]])
                    imgs_candidate_window_pil.append(img_window_pil)
                    if img_window_pil:
                        
                        img_window_cv2 = np.array(img_window_pil.convert('RGB'))[:, :, ::-1].copy()
                        img_window_hsv_cv2 = cv2.cvtColor(img_window_cv2, cv2.COLOR_BGR2HSV)
                        h,s,v = cv2.split(img_window_hsv_cv2)

                        if np.mean(v) < 60:
                            img_window_cv2 = adjust_gamma(img_window_cv2,2)
                        
                        img_window_pil = cv22PIL(img_window_cv2)

                        data = toTensor(img_window_pil)
                        data = data.double()[:3,:,:]
                        data = norm(data)
                        data = data.unsqueeze(0)
                        if self.test_on_gpu:
                            data = data.to(self.device)
                        
                        output = torch.max(self.model(data), 1)  
                        if 1 == int(output[1]):
                            fileName = uuid.uuid4().hex+"_____"+".png"
                            filePath = "./src/construction_site_lane_detection/dataSet/detectedWindows/" + fileName
                            print(filePath)
                            img_window_pil.save(filePath)
                            #print "Detection:: Location -> ({}, {})".format(x, y)
                            detections.append(coordinate_window)
                            scores.append(output[0])
                            cd.append(detections[-1])
                
                detections = torch.tensor(detections).double()
                scores = torch.tensor(scores).double()

                detections_nms_idx = ops.nms(detections,scores,0.2)
                detections_nms = [detections[i] for i in detections_nms_idx]
                print("detected leitbakes=",len(detections_nms))
                if len(detections_nms) >= 3:
                    print("more than 3 leitbakes found!")
                    for img_window_pil in imgs_candidate_window_pil:                 
                        fileName = uuid.uuid4().hex+"_____"+str(idx)+".png"
                        filePath = "./src/construction_site_lane_detection/dataSet/detectedWindows/" + fileName
                        print(filePath)
                        img_window_pil.save(filePath)

            #cv2.rectangle(img_lowerhalf_cv2,(bounding_box.xmin,bounding_box.ymin),(bounding_box.xmax,bounding_box.ymax),(255,0,0),2)
       
            return detections_nms,img_lowerhalf_cv2

        else:

            return None
        
       