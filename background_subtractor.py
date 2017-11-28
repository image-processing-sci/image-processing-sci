import cv2
import os
import numpy as np
from transformation import transform

# Open the video
capture = cv2.VideoCapture('big_files/final.mp4')
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

# if the output mp4 already exists, delete it
try:
    os.remove('outputs/output.mp4')
except:
    pass

# create a new output mp4
video = cv2.VideoWriter('outputs/output.mp4', fourcc, 30.0, size)
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=100, detectShadows=False)
detector = cv2.SimpleBlobDetector_create()

# background image we're doing right
background = cv2.imread('big_files/background.png')
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
FRAMES_FOR_SPEED = 1
SCALING_FACTOR = 1.0

checkerboard_image = cv2.imread('betterCheckb.png')
transformation_matrix, _ = transform(checkerboard_image)

previous_frame_centers = []
frame_count = 0


def calc_euclidean_distance(current_center, previous_center):
    x1, y1 = current_center
    x2, y2 = previous_center
    return ((x1 - x2) ** 2 + (y2 - y1) ** 2) ** 0.5


def match_centers_across_frames(current_frame_centers, previous_frame_centers):
    return_map = dict.fromkeys(range(len(current_frame_centers)))

    parametrized_direction = None

    for i in range(len(current_frame_centers)):
        # start distance at inf
        curr_min_dist = float('inf')
        index_to_pop = None
        for j in range(len(previous_frame_centers)):
            # get euclidean distance
            distance = calc_euclidean_distance(current_frame_centers[i], previous_frame_centers[j])

            if curr_min_dist > distance:
                curr_min_dist = distance
                index_to_pop = j
                x1, y1 = current_frame_centers[i]
                x2, y2 = previous_frame_centers[j]
                parametrized_direction = (x1 - x2, y1 - y2)

        if index_to_pop:
            previous_frame_centers.pop(index_to_pop)
        #TODO: Apply scaling factor
        return_map[i] = (current_frame_centers[i], curr_min_dist * 30.0/FRAMES_FOR_SPEED * SCALING_FACTOR, parametrized_direction)  # this is the speed in pixels per second
    return return_map


def showBirdsEyeCenters(centers):
    # centers[:,2] = 0
    # centers = np.hstack((centers, np.ones((centers.shape[0], 1))))

    output = np.zeros((2000, 2000))
    # apply t_mat to the centers and then plot those onto output
    # import ipdb; ipdb.set_trace()
    centers = cv2.perspectiveTransform(np.array([centers], dtype=float), transformation_matrix)
    # print(centers)

    # import ipdb; ipdb.set_trace()


    for x, y in centers[0]:
        # output[x,y] = 255
        cv2.circle(output, (int(x), int(y)), 20, (255, 255, 255), -1)

    cv2.imshow('birds-eye', output)

    # output = cv2.warpPerspective(img, tMat, (tW, tH))


while True:
    # capture frame in video
    ret, img = capture.read()
    if ret == True:
        if frame_count % FRAMES_FOR_SPEED == 0:
            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # transformed_image = transform(imgray)
            # use the background subtractor
            fgbg.apply(background)
            fgmask = fgbg.apply(imgray)

            # Pre processing, which includes blurring the image and thresholding
            threshold = 30
            new_blur = cv2.GaussianBlur(fgmask, (25, 25), 0)
            ret, thresh = cv2.threshold(new_blur, threshold, 255, cv2.THRESH_BINARY)

            # Get the contours for the thresholded image
            im2, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            blob_area_threshold = 700
            current_frame_centers = []  # will contain a list of the circle coordinates

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

                        centers_xy_coordinates = (cX, cY)
                        if frame_count % FRAMES_FOR_SPEED == 0:
                            current_frame_centers.append(centers_xy_coordinates)

                        # draw the contour and center of the shape on the image
                        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
                        cv2.circle(img, (cX, cY), 7, (0, 255, 0), -1)
            # do this for every FRAMES_FOR_SPEED frames.
            # np_current_frame_centers = np.array(current_frame_centers)
            # np_previous_frame_centers = np.array(previous_frame_centers)
            # np_current_frame_centers.searchsorted(np_previous_frame_centers, side='right')

            current_frame_centers.sort(key=lambda x: -x[1])
            previous_frame_centers.sort(key=lambda x: -x[1])  # sort the centers by closest to camera (bottom of the frame)
            car_map = match_centers_across_frames(current_frame_centers,
                                                  previous_frame_centers)  # need to return velocities of vehicles (speed + direction)
            # print(car_map)
            for key in car_map:
                center, speed, parametrized_direction = car_map[key]
                cX, cY = center
                if speed != float('inf'):
                    cv2.putText(img, str(round(speed)), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(img, str('NaN'), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            previous_frame_centers = current_frame_centers

        cv2.imshow("original footage with blob/centroid", img)

        showBirdsEyeCenters(current_frame_centers)


        frame_count += 1

    if (cv2.waitKey(27) != -1):
        break

capture.release()
video.release()
cv2.destroyAllWindows()