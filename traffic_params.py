import cv2
import os
import numpy as np
from transformation import transform
# from vehicle.py import Car
from car import Car
import ipdb

def calc_euclidean_distance(current_center, previous_center):
    try:
        x1, y1 = current_center
        x2, y2 = previous_center
        return ((x1 - x2) ** 2 + (y2 - y1) ** 2) ** 0.5
    except:
        print('Error')
        import ipdb; ipdb.set_trace()
        return None


def transformToBirdsEye(raw_center, transformation_matrix, preview = False):
    # generate output canvas
    # apply t_mat to the centers
    if not raw_center:
        return []
    transformed_center = cv2.perspectiveTransform(np.array([raw_center], dtype=float), transformation_matrix)
    return transformed_center

def match_center_to_car(transformed_center, visible_cars):
    if not visible_cars:
        return None
    # match transformed xy coordinates, visible_cars
    min_dist = float('inf')
    matched_car_id = None
    for car_id, vehicle in visible_cars.items():

        # haven't used this vehicle
        if not vehicle.updated:
            curr_dist = calc_euclidean_distance(vehicle.get_latest_transformed_center(), transformed_center)
            if curr_dist < min_dist:
                min_dist = curr_dist
                matched_car_id = car_id

    if matched_car_id == None:
        ipdb.set_trace()

    # return the (car_id, vehicle) that matches best
    return (matched_car_id, visible_cars[matched_car_id])

def log_car_details(vehicle, offsets):
    print('visualize {0} and its offsets {1}'.format(vehicle.car_id, offsets))

def log_flow_timestamp(timestamp):
    print('car passed through line at ', timestamp)

def log_density(num_vehicles, timestamp):
    print('{0} cars per 160 ft of two lanes at {1} sec'.format(num_vehicles, timestamp))

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

    FRAMES_PER_SECOND = 30
    FRAMES_FOR_SPEED = 1
    SPEED_SCALING_FACTOR = 0.06818181804 # miles per hour
    LANE_LINES = [880, 1000, 1120]
    LANE_CENTERS = [940, 1060]
    ENTRANCE_RANGE = [200, 250]
    EXIT_RANGE= [1950, 2000]
    BACKGROUND_DIFFERENCE_THRESHOLD = 10
    BLOB_AREA_THRESHOLD = 700 # minimum size of blob in order to be considered a vehicle

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
    num_visible = -1
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
                # use the background subtractor
                fgbg.apply(background)
                fgmask = fgbg.apply(imgray)

                # Pre processing, which includes blurring the image and thresholding

                fgmask = cv2.GaussianBlur(fgmask, (29, 29), 0)
                ret, thresh = cv2.threshold(fgmask, BACKGROUND_DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY)

                if blob_preview: cv2.imshow('blobs', thresh)

                # Get the contours for the thresholded image
                im2, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # loop over the contours
                for c in cnts:
                    area = cv2.contourArea(c)  # getting blob area to threshold
                    # compute the center of the contour
                    if area > BLOB_AREA_THRESHOLD:
                        # import ipdb; ipdb.set_trace()
                        M = cv2.moments(c)
                        # prevent divide by zer0
                        if M["m00"] != 0.0:
                            r_cX = int(M["m10"] / M["m00"])
                            r_cY = int(M["m01"] / M["m00"])

                            raw_center = (r_cX, r_cY)
                            transformed_center = transformToBirdsEye([raw_center], transformation_matrix)
                            t_cX, t_cY = transformed_center[0][0]
                            t_cX += 33 # assume a car is 6.6 feet wide, shift point over by half of the width (3.3 feet = 33 px in transformed plane)

                            if t_cX >= LANE_LINES[0] and t_cX <= LANE_LINES[-1] and t_cY > ENTRANCE_RANGE[0] and t_cY < EXIT_RANGE[1]:
                                if t_cY < ENTRANCE_RANGE[1]:
                                    print("new vehicle", vehicle_id, transformed_center[0][0])
                                    # VEHICLE ENTERED
                                    visible_cars[vehicle_id] = Car(vehicle_id, (r_cX, r_cY), (t_cX, t_cY), c)
                                    vehicle_id += 1
                                    # flow log here
                                    log_flow_timestamp(frame_count / FRAMES_PER_SECOND)

                                elif visible_cars:
                                    # VEHICLE PREVIOUSLY VISIBLE: match with previous entry in visible_cars

                                    car_id, vehicle = match_center_to_car(transformed_center[0][0], visible_cars)
                                    vehicle.updated = True
                                    # add new raw and transformed position
                                    vehicle.update_raw_and_transformed_positions(raw_center, (t_cX, t_cY))
                                    vehicle.update_contour(c)


                # loop through cars to log those which have dissappeared and then remove them
                for car_id, vehicle in visible_cars.items():
                    if not vehicle.updated:
                        vehicle_offsets = [round(min([abs(t_cX - lc) for lc in LANE_CENTERS]) / 10, 1) for (t_cX, t_cY) in vehicle.transformed_centers]
                        log_car_details(vehicle, vehicle_offsets)
                        print('{0} has exited'.format(car_id))
                        # log the car information
                visible_cars = {car_id: vehicle for (car_id, vehicle) in visible_cars.items() if vehicle.updated}

                if len(visible_cars) != num_visible:
                    num_visible = len(visible_cars)
                    log_density(num_visible, frame_count / FRAMES_PER_SECOND)

    # ########

            # display information for cars that are still visible
            for car_id, vehicle in visible_cars.items():
                # reset vehicle to not updated
                vehicle.updated = False
                # raw position, speed and velocity vectors
                current_raw_center = vehicle.get_latest_raw_center()
                speed, previous_raw_center, _ = vehicle.get_latest_raw_velocity()
                if not current_raw_center or not previous_raw_center: continue
                r_cX, r_cY = current_raw_center
                r_pX, r_pY = previous_raw_center
                contour = vehicle.get_latest_contour()

                # transformed position, speed and velocity vectors
                current_transformed_center = vehicle.get_latest_transformed_center()
                _, previous_transformed_center, _ = vehicle.get_latest_transformed_velocity()
                t_cX, t_cY = current_transformed_center
                t_pX, t_pY = previous_transformed_center

                # annotate raw image with contour, centroid
                cv2.drawContours(img, [contour], -1, (0, 0, 204), 2)
                cv2.circle(img, (r_cX, r_cY), 7, (0, 0, 204), -1)
                cv2.putText(img, "{0} mph".format(round(speed, 1)), (r_cX - 20, r_cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 100), 2)
                cv2.arrowedLine(img, (r_pX, r_pY), (r_cX, r_cY), (0,0,100),2)

                if bird_eye_preview:
                    cv2.circle(transformed_output, (int(t_cX), int(t_cY)), 10, (0, 0, 0), -1)
                    cv2.arrowedLine(transformed_output, (int(t_cX), int(t_cY)), (int(t_pX), int(t_pY)), (0,0,0),2)



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