import cv2
import pickle
import numpy as np
from transformation import transform
from car import Car
from graphview.graphview import plot_logs
import argparse

log_attributes = {'num_vehicles': [], 'timestamps': [], 'flow_timestamps': [], 'average_speed': [], 'average_offset': []}

def calc_euclidean_distance(current_center, previous_center):
    x1, y1 = current_center
    x2, y2 = previous_center
    return ((x1 - x2) ** 2 + (y2 - y1) ** 2) ** 0.5


def transformToBirdsEye(raw_center, transformation_matrix, preview = False):
    if not raw_center:
        return []
    # apply t_mat to the centers
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

    # return the (car_id, vehicle) that matches best
    return (matched_car_id, visible_cars[matched_car_id])


def log_flow_timestamp(timestamp):
    log_attributes['flow_timestamps'].append(timestamp)

def log_car_details(vehicle):
    pass

def log_density_and_avg_speed_or_offset(num_vehicles, avg_speed, avg_offset, timestamp):
    log_attributes['num_vehicles'].append(num_vehicles)
    log_attributes['timestamps'].append(timestamp)
    log_attributes['average_speed'].append(avg_speed)
    log_attributes['average_offset'].append(avg_offset)

def main():
    parser = argparse.ArgumentParser()

    # preview settings
    bird_eye_preview = False
    blob_preview = False
    retain_trajectories = False

    parser.add_argument("-bird_eye_preview", action='store_true', help="birds_eye_preview is true")
    parser.add_argument("-blob_preview", action='store_true', help="blob_preview is true")
    parser.add_argument("-retain_trajectories", action='store_true', help="retain_trajectories is true")

    args = parser.parse_args()

    if args.bird_eye_preview:
        bird_eye_preview = True
    if args.blob_preview:
        blob_preview = True
    if args.retain_trajectories:
        retain_trajectories = True

    # Open the video
    capture = cv2.VideoCapture('big_files/final.mp4')

    # background subtraction
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=100, detectShadows=False)

    # background image we're doing right
    background = cv2.imread('big_files/background.png', 0)

    FRAMES_PER_SECOND = 30
    FRAMES_FOR_SPEED = 1
    LANE_LINES = [880, 1000, 1120]
    LANE_CENTERS = [940, 1060]
    ENTRANCE_RANGE = [200, 250]
    EXIT_RANGE= [1950, 2000]
    BACKGROUND_DIFFERENCE_THRESHOLD = 10
    BLOB_AREA_THRESHOLD = 7000  # minimum size of blob in order to be considered a vehicle

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
    frame_count = 0

    visible_cars = {}
    vehicle_id = 0

    # transformed_output = transformed_background
    first_t_plot = True

    # loop through frames of video
    while True:
        # capture current frame in video
        ret, img = capture.read()
        if ret == True:

            if frame_count % FRAMES_FOR_SPEED == 0:

                # birds-eye
                if bird_eye_preview and first_t_plot:
                    transformed_output = transformed_background.copy()
                    if retain_trajectories: first_t_plot = False

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
                cnts = [(c, cv2.contourArea(c)) for c in cnts]
                cnts.sort(key=lambda c: c[1], reverse=True)
                for i, contour_pairing in enumerate(cnts):
                    c, area = contour_pairing
                    # compute the center of the contour
                    if area > BLOB_AREA_THRESHOLD:
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

                                elif visible_cars and i <= len(visible_cars):

                                    # VEHICLE PREVIOUSLY VISIBLE: match with previous entry in visible_cars
                                    car_id, vehicle = match_center_to_car(transformed_center[0][0], visible_cars)
                                    vehicle.updated = True
                                    print('matched updated', car_id)

                                    # add new raw and transformed position
                                    vehicle.update_raw_and_transformed_positions(raw_center, (t_cX, t_cY))
                                    vehicle.update_contour(c)
                    else:
                        # the rest of the blobs do not meet our minimum threshold
                        break

                # loop through cars to log those which have dissappeared and then log/remove them
                for car_id, vehicle in visible_cars.items():
                    if not vehicle.updated:
                        log_car_details(vehicle)
                        print('{0} has exited'.format(car_id))
                        # log the car information
                visible_cars = {car_id: vehicle for (car_id, vehicle) in visible_cars.items() if vehicle.updated}

            speed_sum = 0
            offset_sum = 0
            num_valid_vehicles = 0

            # display information for cars that are still visible
            for car_id, vehicle in visible_cars.items():
                # reset vehicle to not updated
                vehicle.updated = False

                # raw position, speed and velocity vectors
                current_raw_center = vehicle.get_latest_raw_center()
                speed, previous_raw_center, _ = vehicle.get_latest_raw_velocity()
                if not current_raw_center or not previous_raw_center: continue
                num_valid_vehicles += 1
                r_cX, r_cY = current_raw_center
                r_pX, r_pY = previous_raw_center
                contour = vehicle.get_latest_contour()

                # transformed position, speed and velocity vectors
                current_transformed_center = vehicle.get_latest_transformed_center()
                _, previous_transformed_center, _ = vehicle.get_latest_transformed_velocity()
                t_cX, t_cY = current_transformed_center
                t_pX, t_pY = previous_transformed_center

                # update average speed and offset sums
                speed_sum += speed
                offset = min([abs(t_cX - lc) for lc in LANE_CENTERS]) / 10
                offset_sum += offset

                # annotate raw image with contour, centroid
                cv2.drawContours(img, [contour], -1, (0, 0, 204), 2)
                cv2.circle(img, (r_cX, r_cY), 7, (0, 0, 204), -1)
                cv2.putText(img, "{0} mph".format(round(speed, 1)), (r_cX - 20, r_cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 100), 2)
                cv2.arrowedLine(img, (r_pX, r_pY), (r_cX, r_cY), (0,0,100),2)

                if bird_eye_preview:
                    cv2.circle(transformed_output, (int(t_cX), int(t_cY)), 3, (0, 0, 0), -1)
                    cv2.line(transformed_output, (int(t_cX), int(t_cY)), (int(t_pX), int(t_pY)), (0,0,0),2)

            if num_valid_vehicles != 0:
                avg_speed = speed_sum / num_valid_vehicles
                avg_offset = offset_sum / num_valid_vehicles
                log_density_and_avg_speed_or_offset(num_valid_vehicles, avg_speed, avg_offset, frame_count / FRAMES_PER_SECOND)

            cv2.imshow("original footage with blob/centroid", img)
            # birds-eye
            if bird_eye_preview: cv2.imshow('birds-eye', transformed_output)

            frame_count += 1

        if (cv2.waitKey(27) != -1):  # space button
            # save your vars
            pickle.dump(log_attributes, open("log_attributes.p", "wb"))
            plot_logs()
            break

    pickle.dump(log_attributes, open("log_attributes_finished.p", "wb"))
    plot_logs()

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
