import sys
import os

sys.path.insert(0, os.path.abspath("supervision"))
""" The Supervision library is used for object detection and tracking. And this demo contains an edited version of the library to retain information about the original bounding boxes of the detections.
This allows us to visualize the original detections before they were altered by the tracker and kalman filter.
This is useful for visualization and further processing."""
import supervision as sv
from ultralytics import YOLO

import cv2
import numpy as np
from supervision.detection.utils import box_iou_batch

class VehicleDetector:
    """A class to handle vehicle detection in video streams using YOLO and ByteTrack.
    This class initializes the YOLO model, sets up the video capture, and processes frames to detect vehicles.
    It supports Region of Interest (ROI) masking and can handle multiple classes of vehicles.
    Attributes:
        model_path (str): Path to the YOLO model.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        video_path (str): Path to the video file.
        class_ids (list): List of class IDs to detect (default is [2, 3, 5, 7] for car, motorcycle, bus, truck).
        roi_path (str): Path to the ROI mask image (optional).
        start_offset_frames (int): Number of frames to skip at the start of the video (default is 0).
    Methods:
        __init__(model_path, device, video_path, class_ids=None, roi_path=None, start_offset_frames=0):
            Initializes the VehicleDetector with the specified parameters.
        _load_roi(roi_path, video_path):
            Loads the ROI mask from the specified path or derives it from the video path.
        read_frame():
            Reads a frame from the video capture, applying a delay if specified.
        process_frame(frame):
            Processes a single frame to detect vehicles, applying the ROI mask if available.
        release():
            Releases the video capture resource.
        get_current_frame_index():
            Returns the current frame index of the video capture.
    """
    def __init__(self, model_path, device, video_path: str, class_ids=None, roi_path=None, start_offset_frames: int = 0):
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path).to(device)
        self.tracker = sv.ByteTrack()
        self.device = device

        self.class_ids = class_ids if class_ids else [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.class_names = self.model.model.names
        self.conf_threshold = 0.6

        self.roi_mask = self._load_roi(roi_path, video_path)


        self.delay_frames = start_offset_frames
        self.frozen_frame = None
        self.current_frame_index = 0

    def _load_roi(self, roi_path, video_path):
        """Loads the ROI mask from the specified path or derives it from the video path.
        Args:
            roi_path (str): Path to the ROI mask image (optional).
            video_path (str): Path to the video file.
        Returns:
            np.ndarray: The ROI mask as a grayscale image, or None if no ROI is found.
        """
        # If explicitly given, try to load
        if roi_path and os.path.exists(roi_path):
            print(f"Using provided ROI from: {roi_path}")
            return cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)

        # Otherwise try to derive from video path
        base_path, _ = os.path.splitext(video_path)
        auto_roi_path_a = base_path + "_roi.png"
        auto_roi_path_b = base_path + "roi.png"
        auto_roi_path_c = base_path + "-roi.png"
        auto_roi_path_d = base_path + "Roi.png"

        auto_roi_path = auto_roi_path_a if os.path.exists(auto_roi_path_a) else \
                        auto_roi_path_b if os.path.exists(auto_roi_path_b) else \
                        auto_roi_path_c if os.path.exists(auto_roi_path_c) else \
                        auto_roi_path_d if os.path.exists(auto_roi_path_d) else None

        if auto_roi_path:
            print(f"Using auto-detected ROI from: {auto_roi_path}")
            # Ensure the ROI is grayscale
            return cv2.imread(auto_roi_path, cv2.IMREAD_GRAYSCALE)

        print(f" No auto found ROI with filename {auto_roi_path}. Using full frame.")
        return None


    def read_frame(self):
        if self.delay_frames > 0:
            if self.frozen_frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    return False, None
                self.frozen_frame = frame.copy()
            self.delay_frames -= 1
            return True, self.frozen_frame.copy()
        else:
            ret, frame = self.cap.read()
            return ret, frame

    def process_frame(self, frame):
        """Processes a single frame to detect vehicles, applying the ROI mask if available.
        Args:
            frame (np.ndarray): The input frame from the video.
        Returns:
            sv.Detections: The detected vehicles in the frame, with tracking information. Includes original bounding boxes for accurate feature extraction and visualization.
            np.ndarray: The original frame with detections applied.
        Raises:
            ValueError: If the frame is None or empty.
        """

        if self.roi_mask is not None:
            roi_frame = cv2.bitwise_and(frame, frame, mask=self.roi_mask)
        else:
            roi_frame = frame.copy()

        # Get detections from the model
        results = self.model(roi_frame)[0]
        # Convert results to supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # Filter classes & confidence
        detections = detections[np.isin(detections.class_id, self.class_ids)]
        detections = detections[np.greater(detections.confidence, self.conf_threshold)]


        # Update with tracker
        tracked_detections = self.tracker.update_with_detections(detections)


        # If we have detections, we can assign original boxes
        if detections.xyxy.shape[0] > 0 and tracked_detections.xyxy.shape[0] > 0:

            # match tracked boxes with original detections using IoU
            iou_matrix = box_iou_batch(detections.xyxy, tracked_detections.xyxy)

            # Assign each tracked box to the detection with the highest IoU
            best_matches = iou_matrix.argmax(axis=0)  # shape: (num_tracked,)

            # Now for each tracked detection, attach the original box
            # The Original boxes are the ones from the detections before tracking
            # They allow us to get a more accurate representation of the original detection
            # that has not been altered by the tracker and kalman filter
            # This is useful for visualization and further processing
            original_boxes = detections.xyxy[best_matches]

            # Attach to .data["original_xyxy"]
            tracked_detections.data["original_xyxy"] = original_boxes

        else:
            tracked_detections.data["original_xyxy"] = np.empty((0, 4), dtype=np.float32)

        return tracked_detections, frame

    def release(self):
        self.cap.release()

    def get_current_frame_index(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
