from detector import VehicleDetector

from SendDetections import SendDetections

import argparse

import time
import logging
import os
import signal
from metrics import FPSMonitor
from queue import Queue

keep_running = True

fps_monitor = FPSMonitor(window_sec=5)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

### THIS CODE RUNS ON NETWORK CAMERA VIDEO STREAMS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_source', type=str, default='videos/vdo4.avi', help='Path to the first video file. (Re-Identification FROM)')
    # parser.add_argument('--roi_path1', type=str, default="videos/vdo4_roi.png", help='Path to the ROI image for the first video. If not provided, it will try to auto-detect in the same folder based on the video name.')
    parser.add_argument('--roi_path', type=str, help='Path to the ROI image for the first video. If not provided, it will try to auto-detect in the same folder based on the video name.')
    parser.add_argument('--detection_model_path', type=str, default='yolov8x.pt', choices=['yolov8x.pt', 'yolov8l.pt', 'yolov5su.pt'] , help='Path to the YOLO model file.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'], help='Device to run the model on (e.g., "cuda" or "cpu").')

    parser.add_argument('--play_mode', type=int, default=200, help='Delay between frames in milliseconds. Set to 0 for manual frame stepping (Pressing Enter for new frame).')
    
    parser.add_argument('--mqtt_topic', type=str, default="tomass/detections_camera_1", help='mqtt topic to send the detections to.')
    parser.add_argument('--cam_id', type=str, required=True, help='Unique camera identifier (must match feature extractor config)')

    parser.add_argument('--mqtt_broker', type=str, default='reid-vehicle-detection')
    parser.add_argument('--mqtt_port', type=int, default=8884)
    parser.add_argument('--mqtt_certs_path', type=str, default='certs')
    parser.add_argument('--cafile', type=str, default=None)
    parser.add_argument('--certfile', type=str, default=None)
    parser.add_argument('--keyfile', type=str, default=None)



    return parser.parse_args()

def stop(self, signum):
        logging.info(f"\n[INFO] Caught signal {signum}. Exiting gracefully...")
        global keep_running
        keep_running = False  

def run_demo(args):
    logging.info("Starting vehicle detection demo...")

    global keep_running

    # Initialize the vehicle detectors for both videos
    detector = VehicleDetector(
        video_source=args.video_source,
        roi_path=args.roi_path,
        model_path=args.detection_model_path,
        device=args.device
    )

    # Initialize sending class once
    send_detections = SendDetections(
        detector.class_ids,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        mqtt_topic=args.mqtt_topic,
        mqtt_certs_path=args.mqtt_certs_path,
        cafile=args.cafile,
        certfile=args.certfile,
        keyfile=args.keyfile,
        cam_id=args.cam_id
    )

    # ---- MONITORING CONFIG ----
    MAX_FAILURES = 20
    FAILURE_WINDOW = 300  # seconds
    HEARTBEAT_INTERVAL = 30  # seconds

    failure_timestamps = []
    last_heartbeat = time.time()

    while keep_running:

        ret, frame = detector.read_frame()

        # ==========================
        # STREAM FAILURE HANDLING
        # ==========================
        if not ret:
            logging.warning("Frame read failed.")

            now = time.time()
            failure_timestamps.append(now)

            # Keep only recent failures
            failure_timestamps = [
                t for t in failure_timestamps
                if now - t < FAILURE_WINDOW
            ]

            logging.warning(
                f"Failures in last {FAILURE_WINDOW}s: {len(failure_timestamps)}"
            )

            if len(failure_timestamps) > MAX_FAILURES:
                logging.critical("Too many failures. Triggering nuclear exit.")
                os.kill(os.getpid(), signal.SIGTERM)

            continue
        
        # FPS Monitoring
        recv_time = time.time()
        fps_monitor.frame_received()


        # ==========================
        # NORMAL PROCESSING
        # ==========================
        # With FPS monitoring before and after

        detections, frame = detector.process_frame(frame)

        fps_monitor.frame_processed(recv_time)

        send_detections(frame, detections)
        send_detections.clear()

        # ==========================
        # HEARTBEAT LOGGING
        # ==========================
        stats = fps_monitor.get_stats()

        if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
            logging.info("Heartbeat: detection loop alive.")
            last_heartbeat = time.time()

            logging.info(
                f"[FPS] recv_fps={stats['recv_fps']} | "
                f"proc_fps={stats['proc_fps']} | "
                f"latency={stats['avg_latency_ms']} ms"
            )

    logging.info("Shutting down detector.")
    detector.release()

if __name__ == "__main__":
    #run_demo("video1.avi", "video2.avi")
    args = parse_args()

    signal.signal(signal.SIGINT, stop)   # Ctrl+C
    signal.signal(signal.SIGTERM, stop)  # docker stop

    run_demo(args)
