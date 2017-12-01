from background_subtractor import SPEED_SCALING_FACTOR, FRAMES_FOR_SPEED, calc_euclidean_distance


class Car:

    def __init__(self, car_id, raw_center, transformed_center, contour):
        self.transformed_velocities = []  # (speed, previous_position, current_position)
        self.raw_centers = [raw_center]  # (rX, rY)
        self.transformed_centers = [transformed_center]  # (tX, tY)
        self.car_id = car_id  # #
        self.contours = [contour]  # contour object from opencv

    def get_latest_transformed_velocity(self):
        '''
        :return: (speed, previous_center, current_center) tuple of transformed velocities palatable for drawing arrows
        '''
        return self.transformed_velocities[-1] if self.transformed_velocities else (-1, None, None)

    def get_latest_raw_center(self):
        '''
        :return: (x, y) of the latest raw center of a car
        '''
        return self.raw_centers[-1] if self.raw_centers else (-1, -1)

    def get_latest_transformed_center(self):
        '''
        :return: (x, y) of the latest transformed center of a car
        '''
        return self.transformed_centers[-1] if self.transformed_centers else (-1, -1)

    def get_latest_contour(self):
        '''
        :return: OpenCV object containing the latest contour of the car
        '''
        return self.contours[-1] if self.contours else None

    def update_contour(self, contour):
        '''
        :param contour: Newest contour
        :return: None
        '''
        self.contours.append(contour)

    def update_raw_and_transformed_positions(self, raw_center, transformed_center):  # (x, y)
        '''
        Updates raw centers, transformed, centers, and latest transformed velocities
        :param raw_center: (x, y) of the newest raw center
        :param transformed_center: (x, y) of the newest transformed center
        :return: None
        '''
        # add position to positions and add velocities
        transformed_speed = calc_euclidean_distance(self.transformed_centers[-1], transformed_center) * 30.0/FRAMES_FOR_SPEED * SPEED_SCALING_FACTOR # speed in mph
        previous_transformed_center = self.transformed_centers[-1]

        self.transformed_velocities.append((transformed_speed, previous_transformed_center, transformed_center))
        self.transformed_centers.append(transformed_center)
        self.raw_centers.append(raw_center)

    def get_car_id(self):
        return self.car_id

    ### For Logging ###

    def log_details(self):
        '''
        :return: (car_id, transformed_centers (feet), transformed_velocities (mph)) tuple to be used for visualizations.
        All values are in real world units.
        '''
        transformed_centers_in_feet = [center/10 for center in self.transformed_centers]
        return self.car_id, transformed_centers_in_feet, self.transformed_velocities

    def __repr__(self):
        return self.log_details()
