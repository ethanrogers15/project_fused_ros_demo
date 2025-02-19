#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numpy import array
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import cv2
import mediapipe as mp
from numpy.linalg import inv
from mediapipe.tasks.python.components.containers import Detection, DetectionResult, BoundingBox, Category


class Fused_Workflow(Node):
    def __init__(self):
        """Project FUSED Workflow Algorithm
        
        Parameters (Inputs):
            iou_threshold (float): Intersection-over-Union fraction threshold for evaluating bounding box overlaps
            decision_making_mode (str): Which sensors need to agree for the detection to be considered valid? 'all', 'thermal', or 'webcam'
            score_threshold (float): Minimum confidence score that the models must have for the detection to be kept
            max_results (int): Maximum number of detections that may be returned by the models per image
        """
        # Initialize the node
        super().__init__('camera_combiner')
        
        # Read in parameters
        self.declare_parameter('iou_threshold', 0.4)
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.declare_parameter('decision_making_mode', 'thermal')
        self.decision_making_mode = self.get_parameter('decision_making_mode').value
        self.declare_parameter('score_threshold', 0.5)
        score_threshold = self.get_parameter('score_threshold').value
        self.declare_parameter('max_results', 3)
        max_results = self.get_parameter('max_results').value
        
        # Initialize the webcam object detection model
        base_options_webcam = python.BaseOptions(model_asset_path='/project_fused_ros_demo/models/efficientdet_lite0.tflite')
        options_webcam = vision.ObjectDetectorOptions(base_options=base_options_webcam, running_mode=vision.RunningMode.IMAGE, max_results=max_results, score_threshold=score_threshold)
        self.webcam_detector = vision.ObjectDetector.create_from_options(options_webcam)
        
        # Initialize the thermal object detection model
        base_options_thermal = python.BaseOptions(model_asset_path='/project_fused_ros_demo/models/thermal.tflite')
        options_thermal = vision.ObjectDetectorOptions(base_options=base_options_thermal, running_mode=vision.RunningMode.IMAGE, max_results=max_results, score_threshold=score_threshold)
        self.thermal_detector = vision.ObjectDetector.create_from_options(options_thermal)
        
        # Initialize the LiDAR object detection model
        base_options_lidar = python.BaseOptions(model_asset_path='/project_fused_ros_demo/models/lidar.tflite')
        options_lidar = vision.ObjectDetectorOptions(base_options=base_options_lidar, running_mode=vision.RunningMode.IMAGE, max_results=max_results, score_threshold=score_threshold)
        self.lidar_detector = vision.ObjectDetector.create_from_options(options_lidar)
        
        # Set extrinsic translation matrices based on physical measurements, no z translation assumed
        self.T_l2t = array([[1, 0, 0, 0.028],
                            [0, 1, 0, -0.038],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self.T_l2w = array([[1, 0, 0, 0.083],
                            [0, 1, 0, -0.035],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Set extrinsic rotation matrices from stereo calibration
        self.R_t2cₜ = array([[0.804905, 0.593319, 0.010014],
                             [-0.588094, 0.795337, 0.146920],
                             [0.079206, -0.124146, 0.989098]])
        self.R_l2cₜ = array([[0.813639, 0.571181, 0.108367],
                             [-0.580035, 0.784919, 0.217856],
                             [0.039376, -0.240112, 0.969946]])
        self.R_w2cᵣ = array([[0.903012, -0.397065, -0.164039],
                             [0.397183, 0.917127, -0.033513],
                             [0.163751, -0.034891, 0.985884]])
        self.R_l2cᵣ = array([[0.909488, -0.399788, -0.114025],
                             [0.399705, 0.916314, -0.024592],
                             [0.114314, -0.023211, 0.993173]])

        # Set intrinsic matrices for the three sensors
        self.Kₗ = array([[205.046875, 0.0, 107.55435943603516],
                         [0.0, 205.046875, 82.43924713134766],
                         [0.0, 0.0, 1.0]])
        self.Kₜ = array([[161.393925, 0.000000, 78.062273],
                         [0.000000, 161.761028, 59.925115], 
                         [0.000000, 0.000000, 1.000000]])
        self.Kᵣ = array([[446.423112, 0.000000, 163.485603], 
                         [0.000000, 446.765896, 131.217485],
                         [0.000000, 0.000000, 1.000000]])
        
        # Set visualization parameters
        self.text_color = (255, 255, 255)
        self.box_thickness = 3
        self.margin = 5
        self.row_size = -15
        self.font_size = 0.5
        self.font_thickness = 1
        
        # Set CVBRIDGE
        self.bridge = CvBridge()
        
        # Set publishers
        self.lidar_fused_pub = self.create_publisher(Image, '/lidar_fused_output_placeholder', 10)
        self.thermal_fused_pub = self.create_publisher(Image, '/thermal_fused_output_placeholder', 10)
        self.webcam_fused_pub = self.create_publisher(Image, '/webcam_fused_output_placeholder', 10)
        self.lidar_original_pub = self.create_publisher(Image, '/lidar_original_output_placeholder', 10)
        self.thermal_original_pub = self.create_publisher(Image, '/thermal_original_output_placeholder', 10)
        self.webcam_original_pub = self.create_publisher(Image, '/webcam_original_output_placeholder', 10)
        
        # Set subscribers
        self.lidar_sub = Subscriber(self, Image, '/lidar_placeholder')
        self.thermal_sub = Subscriber(self, Image, '/thermal_placeholder')
        self.webcam_sub = Subscriber(self, Image, '/webcam_placeholder')
        
        # Set the synchronizer
        self.sync = ApproximateTimeSynchronizer([self.lidar_sub, 
                                                 self.thermal_sub, 
                                                 self.webcam_sub],
                                                queue_size=10,
                                                slop=0.15)
        
        # Define the callback
        self.sync.registerCallback(self.fuse)
    
    def fuse(self, lidar_msg, thermal_msg, webcam_msg):
        """Main FUSED workflow function: perform fusion based alignment of object detection
        bounding boxes on sychronized LiDAR, thermal, and webcam images for decision making

        Args:
            lidar_msg (ROS2 message): Synchronized LiDAR image message
            thermal_msg (ROS2 message): Synchronized thermal image message
            webcam_msg (ROS2 message): Synchronized webcam image message
        """
        # Convert image messages to CV2
        try:
            lidar_image = self.bridge.imgmsg_to_cv2(lidar_msg, '32FC1')
            thermal_image = self.bridge.imgmsg_to_cv2(thermal_msg, 'mono16')
            webcam_image = self.bridge.imgmsg_to_cv2(webcam_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Error while converting image messages to CV2: {e}')
        
        # Convert LiDAR image to acceptable format
        lidar_image[np.isnan(lidar_image)] = 0
        max_depth = np.max(lidar_image)
        lidar_image_clipped = np.clip(lidar_image, 0, max_depth)
        lidar_image_mm = lidar_image_clipped * 1000
        lidar_image_normalized = cv2.normalize(lidar_image_mm, None, 0, 65535, cv2.NORM_MINMAX)
        lidar_image_8bit = cv2.convertScaleAbs(lidar_image_normalized, alpha=(255.0 / np.max(lidar_image_normalized)))
        lidar_image_equalized = cv2.equalizeHist(lidar_image_8bit)
        # lidar_image_colormap = cv2.applyColorMap(lidar_image_equalized, cv2.COLORMAP_JET)
        lidar_image_rgb = cv2.cvtColor(lidar_image_equalized, cv2.COLOR_GRAY2RGB)
        
        # Convert thermal image to acceptable format
        thermal_image_normalized = cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX)
        thermal_image_8bit = np.uint8(thermal_image_normalized)
        # thermal_image_colormap = cv2.applyColorMap(thermal_image_8bit, cv2.COLORMAP_HOT)
        thermal_image_rgb = cv2.cvtColor(thermal_image_8bit, cv2.COLOR_GRAY2RGB)
        
        # Convert webcam_image to acceptable format
        webcam_image_rgb = cv2.cvtColor(webcam_image, cv2.COLOR_BGR2RGB)
        
        # Perform detection 
        lidar_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=lidar_image_rgb)
        lidar_detection_result = self.lidar_detector.detect(lidar_mp_image)
        thermal_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=thermal_image_rgb)
        thermal_detection_result = self.thermal_detector.detect(thermal_mp_image)
        webcam_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=webcam_image_rgb)
        webcam_detection_result = self.webcam_detector.detect(webcam_mp_image)
        
        # Create copies of untouched images and draw original detections
        lidar_image_equalized_copy = lidar_image_equalized.copy()
        thermal_image_8bit_copy = thermal_image_8bit.copy()
        webcam_image_copy = webcam_image.copy()
        lidar_original_detection_image = self.visualize(lidar_image_equalized, lidar_detection_result)
        thermal_original_detection_image = self.visualize(thermal_image_8bit, thermal_detection_result)
        webcam_original_detection_image = self.visualize(webcam_image, webcam_detection_result)
        
        # Initialize lists for keeping track of detections to be kept out of the next iteration
        thermal_exclude_idx = []
        webcam_exclude_idx = []
        
        # Initialize detection lists for the fused results
        lidar_fused_detections = []
        thermal_fused_detections = []
        webcam_fused_detections = []

        # For loop through each LiDAR detection in the detection result
        if lidar_detection_result.detections:
            for detection in lidar_detection_result.detections:
                if detection.categories[0].category_name != 'Person':
                    continue
                # Define the top left and bottom right points of the detection
                bbox = detection.bounding_box
                x1, y1 = bbox.origin_x, bbox.origin_y # Top left
                x2, y2 = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height # Bottom right

                # Find the depth on the LiDAR image at the center of the box
                uₗ = round((x1 + x2) / 2)
                vₗ = round((y1 + y2) / 2)
                try:
                    zₗ = lidar_image[vₗ,uₗ]
                except IndexError:
                    if uₗ >= lidar_image.shape[1]:
                        uₗ = lidar_image.shape[1] - 1
                    if vₗ >= lidar_image.shape[0]:
                        vₗ = lidar_image.shape[0] - 1
                    zₗ = lidar_image[vₗ, uₗ]

                # If depth is not zero, then compute transformed u and v on webcam and thermal frames
                if zₗ > 1E-3:
                    x1ₗₜ, y1ₗₜ, x1ₗᵣ, y1ₗᵣ = self.transform(zₗ, x1, y1)
                    x2ₗₜ, y2ₗₜ, x2ₗᵣ, y2ₗᵣ = self.transform(zₗ, x2, y2)

                    # Calculate IoU between the mapped bounding box and all detection results from the webcam and thermal images
                    thermal_mapped_box = (x1ₗₜ, y1ₗₜ, x2ₗₜ, y2ₗₜ)
                    if thermal_detection_result.detections and len(thermal_detection_result.detections) != len(thermal_exclude_idx):
                        thermal_ious = []
                        for idxₜ, thermal_detection in enumerate(thermal_detection_result.detections):
                            if thermal_detection.categories[0].category_name != 'Person':
                                thermal_ious.append(0.0)
                                continue
                            if idxₜ in thermal_exclude_idx:
                                thermal_ious.append(0.0)
                                continue
                            thermal_bbox = thermal_detection.bounding_box
                            x1ₜ, y1ₜ = thermal_bbox.origin_x, thermal_bbox.origin_y
                            x2ₜ, y2ₜ = thermal_bbox.origin_x + thermal_bbox.width, thermal_bbox.origin_y + thermal_bbox.height
                            thermal_box = (x1ₜ, y1ₜ, x2ₜ, y2ₜ)
                            thermal_ious.append(self.calc_iou(thermal_box, thermal_mapped_box))

                    webcam_mapped_box = (x1ₗᵣ, y1ₗᵣ, x2ₗᵣ, y2ₗᵣ)
                    if webcam_detection_result.detections and len(webcam_detection_result.detections) != len(webcam_exclude_idx):
                        webcam_ious = []
                        for idxᵣ, webcam_detection in enumerate(webcam_detection_result.detections):
                            if webcam_detection.categories[0].category_name != 'person':
                                webcam_ious.append(0.0)
                                continue
                            if idxᵣ in webcam_exclude_idx:
                                webcam_ious.append(0.0)
                                continue
                            webcam_bbox = webcam_detection.bounding_box
                            x1ᵣ, y1ᵣ = webcam_bbox.origin_x, webcam_bbox.origin_y
                            x2ᵣ, y2ᵣ = webcam_bbox.origin_x + webcam_bbox.width, webcam_bbox.origin_y + webcam_bbox.height
                            webcam_box = (x1ᵣ, y1ᵣ, x2ᵣ, y2ᵣ)
                            webcam_ious.append(self.calc_iou(webcam_box, webcam_mapped_box))

                    # Choose the thermal or webcam detection result corresponding to the LiDAR mapped result whose IoU is the 
                    # largest and also above the defined Combination IoU threshold. In the next iterations of the for loop,
                    # the thermal or webcam detection result that was chosen should not be chosen again to match with another
                    # LiDAR mapped result
                    valid_thermal_iou = None
                    valid_webcam_iou = None
                    if thermal_detection_result.detections and len(thermal_detection_result.detections) != len(thermal_exclude_idx):
                        max_thermal_iou = max(thermal_ious)
                        max_thermal_iou_index = thermal_ious.index(max_thermal_iou)
                        valid_thermal_iou = 0
                        if max_thermal_iou > self.iou_threshold:
                            valid_thermal_iou, valid_thermal_idx = max_thermal_iou, max_thermal_iou_index
                            thermal_exclude_idx.append(valid_thermal_idx)
                    
                    if webcam_detection_result.detections and len(webcam_detection_result.detections) != len(webcam_exclude_idx):
                        max_webcam_iou = max(webcam_ious)
                        max_webcam_iou_index = webcam_ious.index(max_webcam_iou)
                        valid_webcam_iou = 0
                        if max_webcam_iou > self.iou_threshold:
                            valid_webcam_iou, valid_webcam_idx = max_webcam_iou, max_webcam_iou_index
                            webcam_exclude_idx.append(valid_webcam_idx)

                    # Depending on the decision making mode, choose to either keep the mapped result or not based on whether there 
                    # is agreement between all 3 or only two sensors.
                    # If the mapped result is not being kept, then go to the next iteration of the loop. If it is being kept, then
                    # keep the original detections that have been agreed upon according to the decision making mode. For the 
                    # detection that has not been agreed upon, check if it agrees with LiDAR. If it does, keep it. If it does not,
                    # then use the mapped LiDAR detection onto the appropriate camera frame instead.
                    # Store the three fused detection results at each iteration
                    if self.decision_making_mode == 'all':
                        if valid_thermal_iou and valid_webcam_iou:
                            lidar_fused_detections.append(detection)
                            thermal_fused_detections.append(thermal_detection_result.detections[valid_thermal_idx])
                            webcam_fused_detections.append(webcam_detection_result.detections[valid_webcam_idx])
                        else:
                            continue

                    if self.decision_making_mode == 'thermal':
                        if valid_thermal_iou:
                            lidar_fused_detections.append(detection)
                            thermal_fused_detections.append(thermal_detection_result.detections[valid_thermal_idx])
                            if valid_webcam_iou:
                                webcam_fused_detections.append(webcam_detection_result.detections[valid_webcam_idx])
                            else:
                                webcam_fused_detections.append(self.create_detection(detection, webcam_mapped_box))
                        else:
                            continue

                    if self.decision_making_mode == 'webcam':
                        if valid_webcam_iou:
                            lidar_fused_detections.append(detection)
                            webcam_fused_detections.append(webcam_detection_result.detections[valid_webcam_idx])
                            if valid_thermal_iou:
                                thermal_fused_detections.append(thermal_detection_result.detections[valid_thermal_idx])
                            else:
                                thermal_fused_detections.append(self.create_detection(detection, thermal_mapped_box))
                        else:
                            continue
                else:
                    continue

        # With all of the fused detections, create detection results.
        # Draw the bounding boxes on the images corresponding to the results
        if not lidar_fused_detections:
            lidar_fused_image = lidar_image_equalized_copy
        else:
            lidar_fused_detection_result = DetectionResult(detections=lidar_fused_detections)
            lidar_fused_image = self.visualize(lidar_image_equalized_copy, lidar_fused_detection_result)
            
        if not thermal_fused_detections:
            thermal_fused_image = thermal_image_8bit_copy
        else:
            thermal_fused_detection_result = DetectionResult(detections=thermal_fused_detections)
            thermal_fused_image = self.visualize(thermal_image_8bit_copy, thermal_fused_detection_result)
            
        if not webcam_fused_detections:
            webcam_fused_image = webcam_image_copy
        else:
            webcam_fused_detection_result = DetectionResult(detections=webcam_fused_detections)
            webcam_fused_image = self.visualize(webcam_image_copy, webcam_fused_detection_result)
        
        # Publish images with detections
        self.lidar_original_pub.publish(self.bridge.cv2_to_imgmsg(lidar_original_detection_image))
        self.thermal_original_pub.publish(self.bridge.cv2_to_imgmsg(thermal_original_detection_image))
        self.webcam_original_pub.publish(self.bridge.cv2_to_imgmsg(webcam_original_detection_image))
        self.lidar_fused_pub.publish(self.bridge.cv2_to_imgmsg(lidar_fused_image))
        self.thermal_fused_pub.publish(self.bridge.cv2_to_imgmsg(thermal_fused_image))
        self.webcam_fused_pub.publish(self.bridge.cv2_to_imgmsg(webcam_fused_image))
    
    def visualize(self, image, detection_result):
        """Draw bounding boxes on OpenCV images

        Args:
            image (OpenCV image): OpenCV image that the box must be drawn on
            detection_result (MediaPipe detection result): MediaPipe detection result containing bounding box coordinates and labels

        Returns:
            image (OpenCV image): OpenCV image with the boxes and labels drawn
        """
        # Start for loop for all detections
        for detection in detection_result.detections:
            # Draw the bounding box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, self.text_color, self.box_thickness)

            # Write the label
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (self.margin + bbox.origin_x,
                                self.margin + self.row_size + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                        self.font_size, self.text_color, self.font_thickness, cv2.LINE_AA)
            
        return image
    
    def transform(self, zₗ, uₗ, vₗ):
        """Perform transformations to map a pixel from the LiDAR's camera frame onto the thermal and webcam camera frames

        Args:
            zₗ (float): Depth of the pixel, in meters
            uₗ (int): LiDAR pixel coordinate on the x axis
            vₗ (int): LiDAR pixel coordinate on the y axis

        Returns:
            uₜ, vₜ, uᵣ, vᵣ (tuple): Thermal and webcam pixel coordinates, respectively
        """
        # Calculate the 3D physical coordinate of the center of the LiDAR image
        pₗ = array([uₗ, vₗ, 1])
        l̂ₗ = inv(self.Kₗ) @ pₗ
        r̄ₗ = zₗ * l̂ₗ
        
        # Perform extrinsic translations to the thermal sensor and webcam
        r̄ₜ = (inv(self.R_t2cₜ) @ (self.R_l2cₜ @ r̄ₗ)) + array([self.T_l2t[0, 3], self.T_l2t[1, 3], 0]).T
        r̄ᵣ = (inv(self.R_w2cᵣ) @ (self.R_l2cᵣ @ r̄ₗ)) + array([self.T_l2w[0, 3], self.T_l2w[1, 3], 0]).T
        
        # Transform 3D coordinate to thermal and webcam pixel coordinates
        r̃ₜ = array([r̄ₜ[0]/r̄ₜ[2], r̄ₜ[1]/r̄ₜ[2], r̄ₜ[2]/r̄ₜ[2]])
        r̃ᵣ = array([r̄ᵣ[0]/r̄ᵣ[2], r̄ᵣ[1]/r̄ᵣ[2], r̄ᵣ[2]/r̄ᵣ[2]])
        pₜ = self.Kₜ @ r̃ₜ
        pᵣ = self.Kᵣ @ r̃ᵣ
        uₜ, vₜ = pₜ[0], pₜ[1]
        uᵣ, vᵣ = pᵣ[0], pᵣ[1]
        
        return uₜ, vₜ, uᵣ, vᵣ
    
    def calc_iou(self, box_1, box_2):
        """Calculate the Intersection-over-Union between two bounding boxes

        Args:
            box_1 (tuple): Tuple of top-left and bottom-right pixel coordinates for the first bounding box
            box_2 (tuple): Tuple of top-left and bottom-right pixel coordinates for the second bounding box

        Returns:
            iou (float): Intersection-over-Union ratio
        """
        # Get corner values from both boxes
        x1, y1, x2, y2 = box_1
        x3, y3, x4, y4 = box_2
        
        # Get corner values for the intersection box
        x_inter1 = max(x1, x3)
        y_inter1 = max(y1, y3)
        x_inter2 = min(x2, x4)
        y_inter2 = min(y2, y4)
        
        # Calculate the area of the intersection box
        width_inter = max(0, x_inter2 - x_inter1)
        height_inter = max(0, y_inter2 - y_inter1)
        area_inter = width_inter * height_inter
        
        # Calculate the areas of the two boxes
        width_box1 = x2 - x1
        height_box1 = y2 - y1
        width_box2 = x4 - x3
        height_box2 = y4 - y3
        area_box1 = width_box1 * height_box1
        area_box2 = width_box2 * height_box2
        
        # Calculate the area of the full union of the two boxes
        area_union = area_box1 + area_box2 - area_inter
        
        # If union area is zero, return 0
        if area_union == 0:
            return 0.0
        
        # Calculate the IoU
        iou = area_inter / area_union

        return iou
    
    def create_detection(self, lidar_detection, other_detection_box):
        """Create a MediaPipe detection object

        Args:
            lidar_detection (detection object): Original LiDAR MediaPipe detection object
            other_detection_box (tuple): Tuple with bounding box coordinates for mapped LiDAR box onto either
            webcam or thermal camera frames

        Returns:
            detection (detection object): MediaPipe detection object for the mapped LiDAR box onto one of the 
            two other camera frames
        """
        # Get bounding box coordinates and score
        x1, y1, x2, y2 = other_detection_box
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        score = lidar_detection.categories[0].score
        
        # Define data dictionary
        data = {
            "bounding_box": (x1, y1, x2 - x1, y2 - y1),
            "score": score,
            "category_name": "Person"
        }
        
        # Use MediaPipe functions to build the detection object
        bounding_box = BoundingBox(
            origin_x=data["bounding_box"][0],
            origin_y=data["bounding_box"][1],
            width=data["bounding_box"][2],
            height=data["bounding_box"][3]
        )
        
        category = Category(
            index=None, # Optional
            score=data["score"],
            display_name=None, # Optional
            category_name=data["category_name"]
        )
        
        detection = Detection(
            bounding_box=bounding_box,
            categories=[category],
            keypoints=[] # Optional
        )

        return detection
            

def main():
    rclpy.init()
    node = Fused_Workflow()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()