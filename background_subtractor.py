import cv2
import cv2 as cv
import numpy as np
import os
import imutils

# Open the video
capture = cv2.VideoCapture('./final.mp4')
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

# if the output mp4 already exists, delete it
try:
    os.remove('./output.mp4')
except:
    pass

# create a new output mp4
video = cv2.VideoWriter('output.mp4',fourcc, 30.0,size)
fgbg= cv2.createBackgroundSubtractorMOG2(varThreshold = 100, detectShadows = False)
detector = cv2.SimpleBlobDetector_create()

# background image we're doing right
background = cv2.imread('./background.png')
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

while True:
    # capture frame in video
    ret, img = capture.read()
    if ret==True:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # use the background subtractor
        fgbg.apply(background)
        fgmask = fgbg.apply(imgray)

        # Pre processing, which includes blurring the image and thresholding
        threshold = 30
        new_blur = cv2.GaussianBlur(fgmask,(25,25),0)
        ret,thresh = cv2.threshold(new_blur, threshold, 255, cv2.THRESH_BINARY)

        # Get the contours for the thresholded image
        im2, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blob_area_threshold = 700
        # loop over the contours
        for c in cnts:
            area = cv2.contourArea(c)  # getting blob area to threshold
            # compute the center of the contour
            if area > blob_area_threshold:
                M = cv2.moments(c)
                # prevent divide by zer0
                if M["m00"] != 0.0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # draw the contour and center of the shape on the image
                    cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
                    cv2.circle(img, (cX, cY), 7, (0, 255, 0), -1)
                    cv2.putText(img, "center", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.imshow("test", img)

    if(cv2.waitKey(27)!=-1):
        break

capture.release()
video.release()
cv2.destroyAllWindows()
