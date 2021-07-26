import numpy as np
import cv2 as cv
import argparse
from numpy.core.fromnumeric import size
import winsound
import urllib.request  

#Take first frame > detect some Shi-Tomasi corner points in it > track those points using Lucas-Kanade flow
# display/ sound out alert when exceed delta threshold
 
# parameters for ShiTomasi corner detection; detect corner points & track it iteratively using KLT
#image	Input 8-bit or floating-point 32-bit, single-channel image
#corners	Output vector of detected corners.
#maxCorners	Maximum number of corners to return. 
#qualityLevel Parameter characterizing the minimal accepted quality of image corners.
#minDistance	Minimum possible Euclidean distance between the returned corners.
#blockSize	Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 10,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
# https://justinshenk.github.io/posts/2018/04/optical-flow/

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Creating some random colors for the tracks
color = np.random.randint(0,255,(999,3))

# Creating output video
fourcc = cv.VideoWriter_fourcc('X','V','I','D')
out = cv.VideoWriter('Output.mp4', fourcc, 30.0, (800,600))

# capturing input video
cap = cv.VideoCapture('v3.mp4')
#cap = cv.VideoCapture('rtsp://192.168.1.32:8080/h264_ulaw.sdp')
#cap = cv.VideoCapture(0)

# Taking first frame and graying it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# Deciding on the points to track on the first frame; image should be a grayscaled
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params) 

ret,frame = cap.read()
frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# For cv2.calcOpticalFlowPyrLK(), we pass the previous frame, previous points and next frame.
# calculate optical flow
p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

if p1 is not None:
    new_new = p1[st==1]
    old_old = p0[st==1]

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# def combine_two_color_images(image1, image2):

#     foreground, background = image1.copy(), image2.copy()
    
#     foreground_height = foreground.shape[0]
#     foreground_width = foreground.shape[1]
#     alpha = 0.5
    
#     # do composite on the upper-left corner of the background image.
#     blended_portion = cv.addWeighted(foreground,
#                 alpha,
#                 background[:foreground_height,:foreground_width,:],
#                 1 - alpha,
#                 0,
#                 background)
#     background[:foreground_height,:foreground_width,:] = blended_portion
#     cv.imshow('composited image', background)

#     cv.waitKey(10000)

while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    new_canny = cv.Canny(frame_gray, 100, 200) 
    new_canny = cv.cvtColor(new_canny, cv.COLOR_GRAY2BGR) ##2D back to 3D

    # For cv2.calcOpticalFlowPyrLK(), we pass the previous frame, previous points and next frame.
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # print("p1: {}".format(p1))
    # print("st: {}".format(st.shape))

    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    # drawing the tracks
    for i,(new,old,oldest) in enumerate(zip(good_new, good_old, old_old)):

        a,b = new.ravel()
        c,d = old.ravel()
        e,f = oldest.ravel()

        if abs(float(a-e)) > 5 or abs(float(b-f)) > 5:
            mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
            frame = cv.circle(new_canny, (int(a), int(b)), 7, color[i].tolist(), -1)
            frame = cv.circle(new_canny, (int(e), int(f)), 7, color[i].tolist(), -1)

        text = str('X :' + str(format((a-e), ".1f")) + ', Y:' + str(format((b-f), ".1f")))

        # if there is pixel movement above the given threshold, distance will be shown
        if abs(float(a-e)) > 5:
            cv.putText(frame, text, (int(a) -10 ,int(b) -10 ), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if abs(float(b-f)) > 5:
            cv.putText(frame, text, (int(a) -10 ,int(b) -10 ), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    img = cv.add(new_canny,mask) ############################################################
    # print("img type, " +  str(type(img)))
    # print("img size, " +  str(img.shape))
    # print("new_canny type, " +  str(type(new_canny)))
    # print("new_canny size, " +  str(new_canny.shape))
    # print("mask type, " +  str(type(mask)))
    # print("mask size, " +  str(mask.shape))


    img = cv.resize(img,(800,600))

    #produce camera.mp4
    out.write(img) 

    #display output video
    cv.imshow('Video',img)

    # exit player and update reference frame
    k = cv.waitKey(30) & 0xff

    if k == ord('x'):
        break

    if k == ord('k'): 
        mask = np.zeros_like(frame)
        cv.putText(img, "Status: {}".format('Frame captured'), (10, 50), cv.FONT_HERSHEY_COMPLEX, 1, (139, 214, 177), 3)

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2) ### MISSING FROM MAIN.PY