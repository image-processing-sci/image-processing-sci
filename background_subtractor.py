import cv2
import os
import numpy as np
from transformation import transform

def calc_euclidean_distance(current_center, previous_center):
    try:
        x1, y1 = current_center
        x2, y2 = previous_center
        return ((x1 - x2) ** 2 + (y2 - y1) ** 2) ** 0.5
    except:
        import ipdb; ipdb.set_trace()
        return None



def match_centers_across_frames(raw_current_frame_centers, raw_previous_frame_centers, transformed_current_frame_centers, transformed_previous_frame_centers, FRAMES_FOR_SPEED, SPEED_SCALING_FACTOR):
    # import ipdb; ipdb.set_trace()
    if len(transformed_current_frame_centers) == 0 or len(transformed_previous_frame_centers) == 0 or len(raw_current_frame_centers) == 0:
        return {}


    numCurrent = len(transformed_current_frame_centers[0])
    numPrev = len(transformed_previous_frame_centers[0])
    center_correspondence_map = dict.fromkeys(range(numCurrent))
    exhausted_centers = set([])
    raw_parametrized_direction = None
    transformed_parametrized_direction = None

    for i in range(numCurrent):
        # start distance at inf
        curr_min_dist = float('inf')
        exhausted_center_index = None
        for j in range(numPrev):
            if j != exhausted_center_index:
                # get euclidean distance
                distance = calc_euclidean_distance(transformed_current_frame_centers[0][i], transformed_previous_frame_centers[0][j])

                if curr_min_dist > distance:
                    curr_min_dist = distance
                    exhausted_center_index = j

                    xRc, yRc = raw_current_frame_centers[0][i]
                    xRp, yRp = raw_previous_frame_centers[0][j]
                    raw_parametrized_direction = (xRc - xRp, yRc - yRp)

                    xTc, yTc = transformed_current_frame_centers[0][i]
                    xTp, yTp = transformed_previous_frame_centers[0][j]
                    transformed_parametrized_direction = (xTc - xTp, yTc - yTp)


        #TODO: Apply scaling factor to adjust speed
        center_correspondence_map[i] = (raw_current_frame_centers[0][i], transformed_current_frame_centers[0][i], curr_min_dist * 30.0/FRAMES_FOR_SPEED * SPEED_SCALING_FACTOR, raw_parametrized_direction, transformed_parametrized_direction)  # this is the speed in pixels per second

        # remove the center from the remaining previous_frame_center candidates
        if exhausted_center_index:
            # transformed_previous_frame_centers.pop(exhausted_center_index)
            exhausted_centers.add(exhausted_center_index)

    return center_correspondence_map


def transformToBirdsEye(raw_centers, transformation_matrix, preview = False):
    # generate output canvas
    # apply t_mat to the centers
    if not raw_centers:
        return []
    transformed_centers = cv2.perspectiveTransform(np.array([raw_centers], dtype=float), transformation_matrix)
    return transformed_centers

    # if preview:
    #     # draw transformed centroids onto cangas
    #     output = np.zeros((2000, 2000))
    #     for x, y in centers[0]:
    #         cv2.circle(output, (int(x), int(y)), 20, (255, 255, 255), -1)
    #     cv2.imshow('birds-eye', output)


def main():

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
    background = cv2.imread('big_files/background.png', 0)
    # background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    FRAMES_FOR_SPEED = 1
    SPEED_SCALING_FACTOR = 0.06818181804 # miles per hour

    # open transformation calibration checkerboard image
    checkerboard_image = cv2.imread('betterCheckb.png')
    # calculate transformation matrix
    transformation_matrix, _ = transform(checkerboard_image)

    # keep a cache of the previous frame centers
    transformed_previous_frame_centers = []
    raw_previous_frame_centers = []
    frame_count = 0

    # loop through frames of video
    while True:
        # capture current frame in video
        ret, img = capture.read()
        if ret == True:
            if frame_count % FRAMES_FOR_SPEED == 0:
                imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # transformed_image = transform(imgray)
                # use the background subtractor
                fgmask = fgbg.apply(imgray)

                # Pre processing, which includes blurring the image and thresholding
                threshold = 30
                fgmask = cv2.GaussianBlur(fgmask, (25, 25), 0)
                ret, thresh = cv2.threshold(fgmask, threshold, 255, cv2.THRESH_BINARY)

                # Get the contours for the thresholded image
                im2, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                blob_area_threshold = 700 # minimum size of blob in order to be considered a vehicle
                raw_current_frame_centers = []  # will contain a list of the centroids of moving vehicles on raw footage

                # import ipdb; ipdb.set_trace()

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
                                raw_current_frame_centers.append(centers_xy_coordinates)

                            # draw the contour and center of the shape on the image
                            cv2.drawContours(img, [c], -1, (0, 0, 204), 2)
                            cv2.circle(img, (cX, cY), 7, (0, 0, 204), -1)
                    transformed_current_frame_centers = transformToBirdsEye(raw_current_frame_centers, transformation_matrix)
                # do this for every FRAMES_FOR_SPEED frames.

                # current_frame_centers.sort(key=lambda x: -x[1])
                # previous_frame_centers.sort(key=lambda x: -x[1])  # sort the centers by closest to camera (bottom of the frame)
                car_map = match_centers_across_frames([raw_current_frame_centers],
                                                    [raw_previous_frame_centers],
                                                    transformed_current_frame_centers,
                                                    transformed_previous_frame_centers,
                                                    FRAMES_FOR_SPEED,
                                                    SPEED_SCALING_FACTOR)  # need to return velocities of vehicles (speed + direction)

                # put velocities on the original image
                for key in car_map:
                    raw_center, transformed_center, speed, raw_parametrized_direction, transformed_parametrized_direction = car_map[key]
                    cX, cY = raw_center
                    p_dX, p_dY = raw_parametrized_direction

                    if speed != float('inf'):
                        cv2.putText(img, "{0} mph".format(round(speed)), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 100), 2)
                        try:
                            cv2.arrowedLine(img, (cX, cY), (int(cX + p_dX), int(cY + p_dY)), (0,0,100),2)
                        except:
                            import ipdb; ipdb.set_trace()
                        # cv2.line(img, (cX, cY), (cX + p_dX, cY + p_dY), (0,0,255),4)
                    else:
                        cv2.putText(img, str('NaN'), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 100), 2)
                transformed_previous_frame_centers = transformed_current_frame_centers
                raw_previous_frame_centers = raw_current_frame_centers

            cv2.imshow("original footage with blob/centroid", img)

            # transformToBirdsEye(current_frame_centers, transformation_matrix, preview = True)


            frame_count += 1

        if (cv2.waitKey(27) != -1):
            break

    capture.release()
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
