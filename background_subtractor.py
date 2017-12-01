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

        if raw_parametrized_direction and transformed_parametrized_direction:
            #TODO: Apply scaling factor to adjust speed
            center_correspondence_map[i] = (raw_current_frame_centers[0][i], transformed_current_frame_centers[0][i], curr_min_dist * 30.0/FRAMES_FOR_SPEED * SPEED_SCALING_FACTOR, raw_parametrized_direction, transformed_parametrized_direction)  # this is the speed in pixels per second

        # remove the center from the remaining previous_frame_center candidates
        if exhausted_center_index:
            # transformed_previous_frame_centers.pop(exhausted_center_index)
            exhausted_centers.add(exhausted_center_index)

    return center_correspondence_map


def transformToBirdsEye(raw_center, transformation_matrix, preview = False):
    # generate output canvas
    # apply t_mat to the centers
    if not raw_center:
        return []
    transformed_center = cv2.perspectiveTransform(np.array([raw_center], dtype=float), transformation_matrix)
    return transformed_center

def match_center_to_car(transformed_center, visible_cars):
    # match transformed xy coordinates, visible_cars
    min_dist = float('inf')
    for car_id, car in visible_cars.items():
        curr_dist = calc_euclidean_distance(car.get_latest_transformed_center(), transformed_center)
        if curr_dist < min_dist:


    # return the (car_id, car) that matches best

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

    FRAMES_FOR_SPEED = 1
    SPEED_SCALING_FACTOR = 0.06818181804 # miles per hour
    LANE_LINES = [880, 1000, 1120]
    ENTRANCE_RANGE = [200, 250]
    EXIT_RANGE= [1750, 1800]


    # open transformation calibration checkerboard image
    checkerboard_image = cv2.imread('betterCheckb.png')
    # calculate transformation matrix
    transformation_matrix, _ = transform(checkerboard_image)
    transformed_background = cv2.warpPerspective(background, transformation_matrix, (2000, 2000))
    # draw lane lines on background
    for l in LANE_LINES:
        cv2.line(transformed_background, (l, 0), (l, 2000), (0, 0, 0), 3)
    for hg in ENTRANCE_RANGE + EXIT_RANGE:
        cv2.line(transformed_background, (0, hg), (2000, hg), (0, 0, 0), 3)

    # keep a cache of the previous frame centers
    transformed_previous_frame_centers = []
    raw_previous_frame_centers = []
    frame_count = 0

    # preview settings
    bird_eye_preview = True
    blob_preview = False

    visible_cars = {}
    vehicle_id = 0
    log_object = None

    # loop through frames of video
    while True:
        # capture current frame in video
        ret, img = capture.read()
        if ret == True:

            if frame_count % FRAMES_FOR_SPEED == 0:

                # birds-eye
                if bird_eye_preview: transformed_output = transformed_background.copy()

                imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # transformed_image = transform(imgray)
                # use the background subtractor
                fgbg.apply(background)
                fgmask = fgbg.apply(imgray)

                # Pre processing, which includes blurring the image and thresholding
                threshold = 10

                fgmask = cv2.GaussianBlur(fgmask, (29, 29), 0)
                ret, thresh = cv2.threshold(fgmask, threshold, 255, cv2.THRESH_BINARY)

                if blob_preview: cv2.imshow('blobs', thresh)

                # Get the contours for the thresholded image
                im2, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                blob_area_threshold = 700 # minimum size of blob in order to be considered a vehicle
                raw_current_frame_centers = []  # will contain a list of the centroids of moving vehicles on raw footage
                transformed_current_frame_centers = []

                # loop over the contours
                for c in cnts:
                    area = cv2.contourArea(c)  # getting blob area to threshold
                    # compute the center of the contour
                    if area > blob_area_threshold:
                        # import ipdb; ipdb.set_trace()
                        M = cv2.moments(c)
                        # prevent divide by zer0
                        if M["m00"] != 0.0:
                            r_cX = int(M["m10"] / M["m00"])
                            r_cY = int(M["m01"] / M["m00"])

                            raw_center = (r_cX, r_cY)
                            transformed_center = transformToBirdsEye([raw_center], transformation_matrix)
                            # import ipdb; ipdb.set_trace()
                            t_cX, t_cY = transformed_center[0][0]

                            if t_cX >= LANE_LINES[0] and t_cX <= LANE_LINES[-1] and t_cY > ENTRANCE_RANGE[0] and t_cY < EXIT_RANGE[1]:
                                if t_cY < ENTRANCE_RANGE[1]:
                                    # VEHICLE ENTERED
                                    visible_cars[vehicle_id] = Car(vehicle_id, (r_cX, r_cY), (t_cX, t_cY), c)
                                    vehicle_id += 1
                                else:
                                    # VEHICLE PREVIOUSLY VISIBLE
                                    # match with previous entry in visible_cars
                                    car_id, car = match_center_to_car(transformed_center, visible_cars)
                                    # add new raw and transformed position
                                    car.update_raw_and_transformed_position(raw_center, transformed_center)
                                    car.update_contour(c)

                                    if t_cY > EXIT_RANGE[0]:
                                        # VEHICLE EXITING:
                                        # log it and remove from visible cars
                                        log_vehicle(car, log_object)
                                        del visible_cars[car_id]
                                        # del car


# ###################################################
                                # TODO: delete current behavior for all vehicles
                                raw_current_frame_centers.append(raw_center)
                                transformed_current_frame_centers.append([t_cX, t_cY])

                                # draw the contour and center of the shape on the image
                                cv2.drawContours(img, [c], -1, (0, 0, 204), 2)
                                cv2.circle(img, (r_cX, r_cY), 7, (0, 0, 204), -1)

                                # separate into new vehicles
                transformed_current_frame_centers = np.array([transformed_current_frame_centers])

                # birds-eye
                if bird_eye_preview and len(transformed_current_frame_centers) > 0:
                    for x, y in transformed_current_frame_centers[0]:
                        cv2.circle(transformed_output, (int(x), int(y)), 10, (0, 0, 0), -1)

                car_map = match_centers_across_frames([raw_current_frame_centers],
                                                    [raw_previous_frame_centers],
                                                    transformed_current_frame_centers,
                                                    transformed_previous_frame_centers,
                                                    FRAMES_FOR_SPEED,
                                                    SPEED_SCALING_FACTOR)  # need to return velocities of vehicles (speed + direction)

                # put velocities on the original image
                for key in car_map:
                    if car_map[key] == None: continue
                    raw_center, transformed_center, speed, raw_parametrized_direction, transformed_parametrized_direction = car_map[key]
                    r_cX, r_cY = raw_center
                    r_Dx, r_Dy = raw_parametrized_direction
                    t_cX, t_cY = transformed_center
                    t_Dx, t_Dy = transformed_parametrized_direction

                    if speed != float('inf'):

                        cv2.putText(img, "{0} mph".format(round(speed)), (r_cX - 20, r_cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 100), 2)
                        cv2.arrowedLine(img, (r_cX, r_cY), (int(r_cX + r_Dx), int(r_cY + r_Dy)), (0,0,100),2)
                        # birds-eye
                        if bird_eye_preview: cv2.arrowedLine(transformed_output, (int(t_cX), int(t_cY)), (int(t_cX + t_Dx), int(t_cY + t_Dy)), (0,0,0),2)

                transformed_previous_frame_centers = transformed_current_frame_centers
                raw_previous_frame_centers = raw_current_frame_centers
# #######################################################

            for car_id, car in visible_cars.items():
                r_cX, r_cY = car.get_latest_raw_center()
                speed, previous_transformed_center, current_transformed_center = car.get_latest_transformed_velocity()
                t_cX, t_cY = current_transformed_center
                # speed and raw velocity vectors
                # transformed velocity vectors


                if bird_eye_preview: cv2.circle(transformed_output, (int(t_cX), int(t_cY)), 10, (0, 0, 0), -1)


            cv2.imshow("original footage with blob/centroid", img)
            # birds-eye
            if bird_eye_preview: cv2.imshow('birds-eye', transformed_output)

            frame_count += 1

        if (cv2.waitKey(27) != -1):
            break

    capture.release()
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
