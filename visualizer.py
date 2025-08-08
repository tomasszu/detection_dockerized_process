import sys
import os

import cv2

sys.path.insert(0, os.path.abspath("supervision"))
""" The Supervision library is used for object detection and tracking. And this demo contains an edited version of the library to retain information about the original bounding boxes of the detections.
This allows us to visualize the original detections before they were altered by the tracker and kalman filter.
This is useful for visualization and further processing."""
import supervision as sv

class Visualizer:
    """A class for visualizing vehicle tracking and ReID results.
    This class provides methods to annotate frames with bounding boxes, labels, and traces of tracked vehicles.
    It uses the supervision library for annotations and supports custom class names and trace drawing.
    """
    def __init__(self, class_names: dict, traces=True, rows=None, cols=None,
                 area_bottom_left = None, area_top_right=None):
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        self.class_names = class_names
        self.draw_traces = traces

        self.rows = rows
        self.cols = cols

        # Set the cropping area
        self.area_bottom_left = area_bottom_left  # (x_min, y_max)
        self.area_top_right = area_top_right      # (x_max, y_min)

        self.zones = self._generate_zones()

    def _generate_zones(self):
        """Generates the zones based on the specified rows and columns within the defined area.
        Returns:
            list: A list of tuples representing the zones, each defined by its top-left and bottom-right coordinates.
        """
        x_min, y_max = self.area_bottom_left
        x_max, y_min = self.area_top_right

        zone_width = (x_max - x_min) / self.cols
        zone_height = (y_max - y_min) / self.rows
        zones = []

        for i in range(self.rows):
            for j in range(self.cols):
                x1 = x_min + j * zone_width
                y1 = y_min + i * zone_height
                x2 = x_min + (j + 1) * zone_width
                y2 = y_min + (i + 1) * zone_height
                zones.append((int(x1), int(y1), int(x2), int(y2)))

        return zones
    
    def draw_debug(self, frame):
        """Draws the zones and center points on the frame for debugging purposes.
        Args:
            frame (np.ndarray): The frame on which to draw the debug information.
            center_points (list): List of center points of the detections to be drawn.
        """
        for zone in self.zones:
            x1, y1, x2, y2 = zone
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        return frame


    def annotate(self, frame, detections):
        labels = []
        for _, _, confidence, class_id, tracker_id, _ in detections:
            if tracker_id == -1:
                label = "Unknown"
            else:
                name = self.class_names.get(class_id, "Vehicle")
                label = f"ID {tracker_id} {name} {confidence:.2f}"
            labels.append(label)

        frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        if self.draw_traces:
            frame = self.trace_annotator.annotate(scene=frame, detections=detections)


        ###--------------------- DEBUG ------------------------------------------------------###
        ### Drawing crop zones


        frame = self.draw_debug(frame)



        ###--------------------- DEBUG END ------------------------------------------------------###

        return frame
