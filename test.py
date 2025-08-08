from detector import VehicleDetector
from visualizer import Visualizer

from SendDetections import SendDetections

import cv2

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path1', type=str, default='videos/vdo4.avi', help='Path to the first video file. (Re-Identification FROM)')
    parser.add_argument('--roi_path1', type=str, default="videos/vdo4_roi.png", help='Path to the ROI image for the first video. If not provided, it will try to auto-detect in the same folder based on the video name.')
    parser.add_argument('--detection_model_path', type=str, default='yolov8x.pt', choices=['yolov8x.pt', 'yolov8l.pt', 'yolov5su.pt'] , help='Path to the YOLO model file.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'], help='Device to run the model on (e.g., "cuda" or "cpu").')

    parser.add_argument('--play_mode', type=int, default=100, help='Delay between frames in milliseconds. Set to 0 for manual frame stepping (Pressing Enter for new frame).')



    return parser.parse_args()

def run_demo(video_path1, roi_path1, detection_model, device, play_mode):
    """ Run the vehicle re-identification demo with two videos. 
    Args:
        video_path1 (str): Path to the first video file.
        video_path2 (str): Path to the second video file.
        roi_path1 (str): Path to the ROI image for the first video.
        roi_path2 (str): Path to the ROI image for the second video.
        detection_model (str): Path to the YOLO model file.
        device (str): Device to run the model on (e.g., "cuda" or "cpu").
        scnd_video_offset_frames (int): Number of frames to delay processing the second video from the start of the first video.
        reID_features_size (int): Size of the feature embeddings to be stored in the database.
        debug (bool): Enable debug mode to visualize crop zones and other debug information.
        features_expire (int): Number of frames after which the feature embeddings will be deleted from the database.
        crop_zone_rows_1 (int): Number of rows in the crop zone grid for the first video.
        crop_zone_cols_1 (int): Number of columns in the crop zone grid for the first video.
        crop_zone_area_bottom_left_1 (tuple): Bottom-left corner of the crop zone area for the first video as a tuple (x, y).
        crop_zone_area_top_right_1 (tuple): Top-right corner of the crop zone area for the first video as a tuple (x, y).
        crop_zone_rows_2 (int): Number of rows in the crop zone grid for the second video.
        crop_zone_cols_2 (int): Number of columns in the crop zone grid for the second video.
        crop_zone_area_bottom_left_2 (tuple): Bottom-left corner of the crop zone area for the second video as a tuple (x, y).
        crop_zone_area_top_right_2 (tuple): Top-right corner of the crop zone area for the second video as a tuple (x, y).
    """
    print("Starting vehicle detection demo...")

    # Initialize the vehicle detectors for both videos
    detector = VehicleDetector(video_path=video_path1, roi_path=roi_path1, model_path=detection_model, device=device)

    # Initialize sending class once
    send_detections = SendDetections(detector.class_ids)

    
    # Initialize the visualizers for both videos
    # The visualizers will annotate the frames with the detections and matched IDs
    visualizer = Visualizer(detector.class_names)

    while True:
        ret1, frame = detector.read_frame()

        if not ret1:
            print("End of video stream.")
            break

        # Process the frames from both videos
        detections, frame = detector.process_frame(frame)


        
        send_detections(frame, detections)

        send_detections.clear()


        #crops = cropper.crop(frame1, detections1)


        
        vis_frame = visualizer.annotate(frame, detections)

        frame = cv2.resize(vis_frame, (1280, 720))


        cv2.imshow("Vehicle Re-ID Demo", frame)
        if cv2.waitKey(play_mode) & 0xFF == ord('q'):
            break

    detector.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #run_demo("video1.avi", "video2.avi")
    args = parse_args()

    run_demo(video_path1=args.video_path1, roi_path1=args.roi_path1, detection_model=args.detection_model_path, device=args.device, play_mode=args.play_mode)
