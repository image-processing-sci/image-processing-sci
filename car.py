import numpy as np

SPEED_SCALING_FACTOR = 0.06818181804
FRAMES_FOR_SPEED = 1
EXIT_LINE = 1800
# FLOW_TRANSFORMED_LINES = np.array([1800, 1300])

class Car:

    def __init__(self, car_id, raw_center, transformed_center, contour):
        self.transformed_velocities = []  # (speed, previous_position, current_position)
        self.raw_velocities = []
        self.raw_centers = [raw_center]  # (rX, rY)
        self.transformed_centers = [transformed_center]  # (tX, tY)
        self.car_id = car_id  # #
        self.contours = [contour]  # contour object from opencv
        self.exited = False

    def get_latest_transformed_velocity(self):
        '''
        :return: (speed, previous_center, current_center) tuple of transformed velocities palatable for drawing arrows
        '''
        return self.transformed_velocities[-1] if self.transformed_velocities else (-1, None, None)

    def get_latest_raw_velocity(self):
        '''
        :return: (speed, previous_center, current_center) tuple of raw velocities palatable for drawing arrows
        '''
        return self.raw_velocities[-1] if self.raw_velocities else (-1, None, None)

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
        real_speed = self._calculate_speed(self.transformed_centers[-1], transformed_center)
        previous_transformed_center = self.transformed_centers[-1]
        previous_raw_center = self.raw_centers[-1]

        self.transformed_velocities.append((real_speed, previous_transformed_center, transformed_center))
        self.raw_velocities.append((real_speed, previous_raw_center, raw_center))

        self.transformed_centers.append(transformed_center)
        self.raw_centers.append(raw_center)

        if transformed_center[1] >= EXIT_LINE:
            # mark as exited
            self.exited = True

    def calc_euclidean_distance(current_center, previous_center):
        try:
            x1, y1 = current_center
            x2, y2 = previous_center
            return ((x1 - x2) ** 2 + (y2 - y1) ** 2) ** 0.5
        except:
            import ipdb; ipdb.set_trace()
            return None

    def _calculate_speed(self, center_1, center_2):
        return Car.calc_euclidean_distance(center_1, center_2) * 30.0/FRAMES_FOR_SPEED * SPEED_SCALING_FACTOR  # speed in mph

    def get_car_id(self):
        return self.car_id

    ### For Logging ###

    def log_details(self):
        '''
        :return: str((car_id, transformed_centers (feet), transformed_velocities (mph))) tuple to be used for visualizations.
        All values are in real world units.
        '''
        transformed_centers_in_feet = [(x/10, y/10) for x,y in self.transformed_centers]
        return self.car_id, transformed_centers_in_feet, self.transformed_velocities

    def __repr__(self):
        return 'ID: {0}\nPositions\n\t{1}\nVelocities\n\t{2}'.format(self.car_id,
                    [(round(x), round(y))for x,y in self.transformed_centers],
                    [round(v[0]) for v in self.transformed_velocities])

