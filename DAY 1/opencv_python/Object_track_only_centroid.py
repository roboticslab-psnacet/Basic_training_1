import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
##    lower_blue = np.array([110,50,50])
    lower_blue = np.array([90,150,150])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    if len(contours) > 0 :
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0 :
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.circle(res, center, 5, (0, 0, 255), -1)
            
            cv2.drawContours(frame,c,-1,(0, 0, 255),1)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            cv2.line(frame,(center[0],0),(center[0],480),(255,255,255),1) 
            cv2.line(frame,(0,center[1]),(648,center[1]),(255,255,255),1) 
            
            cv2.drawContours(res,c,-1,(0, 0, 255),2)
            cv2.circle(res, center, 5, (0, 255, 0), -1)
            cv2.line(res,(center[0],0),(center[0],480),(255,255,255),1) 
            cv2.line(res,(0,center[1]),(648,center[1]),(255,255,255),1) 

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
