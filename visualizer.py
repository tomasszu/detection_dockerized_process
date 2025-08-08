import sys
import os

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
    def __init__(self, class_names: dict, traces=True):
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        self.class_names = class_names
        self.draw_traces = traces

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

        return frame
