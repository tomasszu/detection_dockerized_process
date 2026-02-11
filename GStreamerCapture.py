import cv2
import logging


class LiveGStreamerCapture:
    """
    Low-latency RTSP capture using GStreamer.
    Keeps only the newest frame (no buffering).
    """

    def __init__(self, rtsp_url: str, latency: int = 0):
        self.rtsp_url = rtsp_url

        gst_pipeline = (
            f"rtspsrc location={rtsp_url} latency={latency} drop-on-latency=true ! "
            f"rtph264depay ! h264parse ! nvv4l2decoder ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=3 sync=false"
        )

        logging.info(f"Opening GStreamer pipeline:\n{gst_pipeline}")

        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open RTSP stream with GStreamer")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    def get(self, prop):
        return self.cap.get(prop)
