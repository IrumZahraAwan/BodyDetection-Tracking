# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import time
import threading
import Queue
import multiprocessing
from multiprocessing import Manager
import Queue
import os
import psutil

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

save_body_path = '/ADD/PATH/HERE'

t=time.time()

webcam , ret, frame , image , orig , rects , weights , body_big = None , None , None , None , None , None , None , None
cur_date , cur_time , new_pin , filename1 , filename2 , sampleFile = None , None , None , None , None , None
check , head ,tail = None , None , None
thread_list = []
thread_count = 0
offline_thread_list = []
offline_thread_count = 0
rectangleColor = (0,165,255)
frame_hist = 0
frameCounter = 0
tracker_count = 0
curr_rect = 0
first = 0
status = None

# Any big enough random values just to make the condition false
t_x = 10001
t_y = 10001
t_w = 10031
t_h = 10031
t_x_bar = 100021
t_y_bar = 100011

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

# Capture Camera Stream
webcam = cv2.VideoCapture(0)

# Set Camera Resolution
#webcam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280)
#webcam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 720)

def body_detection():

    global frame_hist
    global first
    global tracker_count
    global t_x
    global t_y
    global t_w
    global t_h 
    global t_x_bar 
    global t_y_bar
    global output_file_path
    global finalLOG_file_path
    global last_server_resp
    global last_upload
    global save_body_path
    global t


    while True: 

        # read each frame
        ret, frame = webcam.read()

        while ret == False:
            print('Frame ',ret)
            sleep(3)
            ret, frame = webcam.read()

        if frame_hist == 0:
            hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
            roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
            frame_hist = 1

        # resize it
        image = imutils.resize(frame, width=min(300, frame.shape[1]))
        orig = image.copy()

        # detect people in the frame
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
            padding=(0, 0), scale=1.1)

        if len(rects) == 0:
            t_x = 10001
            t_y = 10001
            t_w = 10031
            t_h = 10031
            t_x_bar = 100021
            t_y_bar = 100011

        # draw the original bounding boxes
        for i in range(len(rects)):

            body_i = rects[i]
            (x, y, w, h) = [v * 1 for v in body_i]

            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # apply non-maxima suppression
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

            # draw the final bounding boxes
            for i in range(len(pick)):

                body_i = pick[i]
                (xA, yA, xB, yB) = [int(v * 1) for v in body_i]

                # rect on scaled image
                #cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2) 

                # rects to map on original frame
                (x1, y1, w1, h1) = [int(v * 4.28) for v in body_i]
                cv2.rectangle(frame, (x1, y1), (w1, h1), (0, 255, 55), 2)

                curr_rect = (x1, y1, w1, h1)

                # for first run, set tracking window here
                if first == 0:
                    track_window = curr_rect
                    first = 1

                #calculate the centerpoint of NEW bounding boxes
                x_bar = x1 + 0.5 * w1
                y_bar = y1 + 0.5 * h1

                # CHECK IF CURRENT RECTs LIES SOMWHERE IN THE PREVIOUS RECTs
                if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x1 <= t_x_bar <= (x1 + w1 )) and ( y1 <= t_y_bar <= (y1 + h1  ))):
                    
                    print ('RECT MATCHED - KEEP TRACKING - DONT RESET')
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                    # apply meanshift to get the new location
                    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
                    x3,y3,w3,h3 = track_window
                    x3 = ((x1-x3)+x3)
                    y3 = ((y1-y3)+y3)
                    w3 = ((w1-w3)+w3)
                    h3 = ((h1-h3)+h3)

                    # draw tracking rectangles
                    cv2.rectangle(frame, (x3, y3),(w3, h3),rectangleColor ,2)

                    # copy current rects in tracking rects
                    (t_x , t_y , t_w , t_h) = curr_rect
                    #calculate the centerpoints
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h
                   

                else:
                    #print('NO MATCHING RECTS - UPDATE TRACKER - UPDATE RECTS')
                    # track the current rectangle
                    track_window = curr_rect

                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                    # apply meanshift to get the new location
                    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
                    x3,y3,w3,h3 = track_window
                    x3 = ((x1-x3)+x3)
                    y3 = ((y1-y3)+y3)
                    w3 = ((w1-w3)+w3)
                    h3 = ((h1-h3)+h3)

                    # draw tracking rectangles
                    cv2.rectangle(frame, (x3, y3),(w3, h3),rectangleColor ,2)
                   
                    # copy current rects in tracking rects
                    (t_x , t_y , t_w , t_h) = curr_rect

                    #calculate the centerpoints
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    # Crop body from Original frame
                    body_big = frame[y1:h1, x1:w1]

                    ####################################

                    im_shape = body_big.shape
                    #print('ORIGINAl Width: ',im_shape[0])
                    #print('ORIGINAL Height: ',im_shape[1])
                    aspect_ratio = float(float(im_shape[0]) / float(im_shape[1]))
                    #print('ORIGINAL Aspect Ratio: ',aspect_ratio)
                    ratio_check = float(1 / 1.67)
                    #print ('Aspect Ratio Threshold: ', ratio_check)

                    if aspect_ratio < (ratio_check) or aspect_ratio > (ratio_check):
                        new_width = ratio_check * float(im_shape[1])
                        #print('NEW width: ', new_width)
                        aspect_ratio = float(float(new_width) / float(im_shape[1]))
                        #print('NEW aspect ratio: ', aspect_ratio)
                        body_big = imutils.resize(body_big, width=int(new_width))
                        
                    ####################################
                    # Save body
                
                    cur_date = (time.strftime("%Y-%m-%d"))
                    cur_time = (time.strftime("%H:%M:%S"))
                    new_pin =cur_date+"-"+cur_time
                    filename1 = 'UNKNOWN'
                    filename2 = str(filename1)+'-'+str(new_pin)
                    sampleFile = ('%s/%s.png' % (save_body_path, filename2))

                    #Save image in a folder
                    cv2.imwrite('%s/%s.png' % (save_body_path, filename2), body_big)
 
                
        # show the output images
        #cv2.imshow("Before NMS", orig)
        cv2.imshow("After NMS", image)
        #cv2.imshow("ANZEN", frame)

        key = cv2.waitKey(10)
        if key == 27:
            break
        
if __name__ == '__main__':

    print ('Starting body_detection')
    body_det = multiprocessing.Process(target=body_detection )
    body_det.start()
    
    
    
    





