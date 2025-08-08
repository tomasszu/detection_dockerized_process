from detector import VehicleDetector

from SendDetections import SendDetections

import argparse

import time

import signal

keep_running = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path1', type=str, default='videos/vdo4.avi', help='Path to the first video file. (Re-Identification FROM)')
    parser.add_argument('--roi_path1', type=str, default="videos/vdo4_roi.png", help='Path to the ROI image for the first video. If not provided, it will try to auto-detect in the same folder based on the video name.')
    parser.add_argument('--detection_model_path', type=str, default='yolov8x.pt', choices=['yolov8x.pt', 'yolov8l.pt', 'yolov5su.pt'] , help='Path to the YOLO model file.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'], help='Device to run the model on (e.g., "cuda" or "cpu").')

    parser.add_argument('--play_mode', type=int, default=200, help='Delay between frames in milliseconds. Set to 0 for manual frame stepping (Pressing Enter for new frame).')



    return parser.parse_args()

def stop(self, signum):
        print(f"\n[INFO] Caught signal {signum}. Exiting gracefully...")
        global keep_running
        keep_running = False  

def run_demo(video_path1, roi_path1, detection_model, device, play_mode):
    print("Starting vehicle detection demo...")

    global keep_running

    # Initialize the vehicle detectors for both videos
    detector = VehicleDetector(video_path=video_path1, roi_path=roi_path1, model_path=detection_model, device=device)

    # Initialize sending class once
    send_detections = SendDetections(detector.class_ids)

    while keep_running:
        ret1, frame = detector.read_frame()

        if not ret1:
            print("End of video stream.")
            break

        # Process the frames from both videos
        detections, frame = detector.process_frame(frame)


        
        send_detections(frame, detections)

        send_detections.clear()

        # Add a small pause (e.g. 33ms = ~30 FPS)
        time.sleep(play_mode/1000)

    detector.release()

if __name__ == "__main__":
    #run_demo("video1.avi", "video2.avi")
    args = parse_args()

    signal.signal(signal.SIGINT, stop)   # Ctrl+C
    signal.signal(signal.SIGTERM, stop)  # docker stop

    run_demo(video_path1=args.video_path1, roi_path1=args.roi_path1, detection_model=args.detection_model_path, device=args.device, play_mode=args.play_mode)
