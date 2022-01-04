import cv2
import numpy as np
import math
import yaml
import os
import warnings


class Distrakt:
    """Distrakt is a python library for distance tracking, hence the name.
       It works with both images and videos.

       There are currently two methods for shape detection.
            color: This method finds objects with the given color. Default color is yellow. But can be changed to any color from the configs.
            shape: This method finds objects by their shape. It finds countours that have the specified shape. By default it finds triangles and rectangles.

        There is currently only one distance metric implemented.
            euclidean: It calculates the euclidean distance between the two centroids of detected objects.

    Args:
        file_path (str): The path of the video or image.
        configs_path (str, optional): The path to custom configs. If not passed it will take the default configs from the package. If passed, it will only update the parameters that are passed in the custom config file, the rest would be taken from the default configs. Defaults to None.
    """

    __CENTROID_METHODS = ('color', 'shape')
    __DISTANCE_METHODS = ('euclidean')

    __SUPPORTED_IMAGE_EXTENSIONS = ('.jpeg', '.jpg', '.tif', '.tiff', 'png')
    __SUPPORTED_VIDEO_EXTENSIONS = ('.mp4', '.avi')
    
    def __init__(self, file_path: str, configs_path: str=None):

        if not isinstance(file_path, str):
            raise TypeError(f"Should be of type {type(str)}, but provided {type(file_path)}.")
        
        if not os.path.isfile(file_path):
            raise ValueError("File does not exist.")

        if file_path.endswith(Distrakt.__SUPPORTED_IMAGE_EXTENSIONS):
            self.type = 'image'
        elif file_path.endswith(Distrakt.__SUPPORTED_VIDEO_EXTENSIONS):
            self.type = 'video'
        else:
            raise ValueError(f"File extension not supported. Passed {file_path}. Supported extensions for images are: {Distrakt.__SUPPORTED_IMAGE_EXTENSIONS} and for videos: {Distrakt.__SUPPORTED_VIDEO_EXTENSIONS}.")

        self.file_path = file_path

        self.__process_centroid_map = {
            'color': self._get_color_centroids,
            'shape': self._get_shape_centroids
            }

        self.__process_distance_map = {
            'euclidean': self._calculate_distance_euclidean
            }

        self.__CONFIG_PATH = os.path.normpath(os.path.join(os.path.realpath(__file__), '../../configs.yaml'))

        with open(self.__CONFIG_PATH, 'r') as config_file:
            self.default_configs = yaml.load(config_file, Loader=yaml.FullLoader)

        self.configs = Distrakt._read_config(configs_path)

        self.centroid_method = self.configs.get("centroid_method", self.default_configs.get('centroid_method'))
        
        if self.centroid_method not in Distrakt.__CENTROID_METHODS:
            raise ValueError(f"Must be one of {Distrakt.__CENTROID_METHODS}, but passed {self.centroid_method}.")
        
        self.distance_method = self.configs.get("distance_method", self.default_configs.get('distance_method'))
        
        if self.distance_method not in Distrakt.__DISTANCE_METHODS:
            raise ValueError(f"Must be one of {Distrakt.__DISTANCE_METHODS}, but passed {self.distance_method}.")


    def get_distance(self):
        """The main function to calculate the distance between two objects.
        """

        if self.type == "image":
            frame = self.read_image()
            dist = self._process_frame(frame)

            return dist
        
        elif self.type == "video":
            cap = self.read_video()

            dist_list = []

            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    dist = self._process_frame(frame)
                    dist_list.append(dist)
                else:
                    break
            cap.release()

            return dist_list


    def read_image(self):
        """Reads and returnd an image.
        """
        frame = cv2.imread(self.file_path)
        self.hh, self.ww = frame.shape[:2]

        return frame
    

    def read_video(self):
        """Reads and returns the video.
        """
        cap = cv2.VideoCapture(self.file_path)
        self.hh = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.ww = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        return cap


    @staticmethod
    def _read_config(configs_path: str):
        if configs_path:
            if not isinstance(configs_path, str):
                raise TypeError(f"Should be of type {type(str)}, but provided {type(configs_path)}.")
            
            if not os.path.isfile(configs_path):
                raise ValueError("File does not exist.")

            if not configs_path.endswith('.yaml'):
                raise TypeError("Wrong type of file passed.")
            
            with open(configs_path, 'r') as config_file:
                configs = yaml.load(config_file, Loader=yaml.FullLoader)
        else:
            configs = {}
        
        return configs


    def _process_frame(self, frame):
        """Processes each frame. Is used for both image and video analysis.
        """

        nlabels, stats, centroids = self.__process_centroid_map[self.centroid_method](frame)

        # nlabels, stats, centroids = self._get_color_centroids(frame)

        pts = self._postprocess_centroids(nlabels, stats, centroids)

        if len(pts) != 2:
            warnings.warn(f"Found {len(pts)} objects. Skipping calculating distance")
            dist = np.nan
        
        else:
            dist = self.__process_distance_map[self.distance_method](pts)

        return dist


    def _get_shape_centroids(self, frame):
        """Gets the centroids by using the shape method
        """

        imgBlur = cv2.GaussianBlur(frame, (7, 7), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

        imgCanny = cv2.Canny(imgGray, 10, 20)

        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

        kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        morph = cv2.morphologyEx(imgDil, cv2.MORPH_CLOSE, kernel_2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_2)

        contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        nlabels = 0
        centroids = []
        areas = []

        for contour in contours:
            shape = cv2.approxPolyDP(contour, 0.1*cv2.arcLength(contour, True), True)

            if len(shape) == 3 or len(shape) == 4:

                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                centroid = [cX, cY]

                area = cv2.contourArea(contour)

                nlabels += 1
                centroids.append(centroid)
                areas.append(area)

        return nlabels, areas, centroids


    def _get_color_centroids(self, frame):
        """Gets the centroids by using the color method.
        """

        lower_ = self.default_configs['lower']
        upper_ = self.default_configs['upper']

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_ = np.array(lower_)
        upper_ = np.array(upper_)
        thresh = cv2.inRange(hsv, lower_, upper_)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph, None, None, None, 8, cv2.CV_16U)

        areas = stats[0:, cv2.CC_STAT_AREA]

        return nlabels, areas, centroids


    def _postprocess_centroids(self, nlabels, areas, centroids):
        """Does some post processing of the centroids. Removes the ones that are smaller that 20% of the image.
        """

        pts = []
        for i in range(nlabels):
            if areas[i] <= self.ww*self.hh/5:

                pts.append(centroids[i])
        
        return pts

    
    def _calculate_distance_euclidean(self, pts: list):
        """Calculates the Euclidean distance.
        """

        dist = math.dist(*pts)

        return dist





