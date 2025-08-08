import numpy as np
import cv2
import base64
import json
import paho.mqtt.client as mqtt
import time

class SendDetections:
    def __init__(self, class_ids_of_interest, mqtt_broker="localhost", mqtt_port=1884, mqtt_topic="tomass/detections"):
        self.class_ids = class_ids_of_interest
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic
        self.connected = False  # Flag to track connection status
        self.data = []

        # Setup MQTT client
        self.client = mqtt.Client(client_id="sender1")
        self.client.on_connect = self.on_connect
        self.client.on_publish = self.on_publish

        self.client.connect(self.mqtt_broker, self.mqtt_port)
        self.client.loop_start()  # Start in background

        # Wait until connection is established
        timeout = time.time() + 5  # max 5 seconds wait
        while not self.connected:
            if time.time() > timeout:
                raise TimeoutError("MQTT connection timed out.")
            time.sleep(0.5)


    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("[MQTT] Connected successfully")
            self.connected = True
        else:
            print(f"[MQTT] Connection failed with code {rc}")

    def on_publish(self, client, userdata, mid):
        print("[MQTT] Message published, mid =", mid)

    def __call__(self, frame: np.ndarray, detections):
        """
        Processes detections: crops objects of interest and stores them.

        Parameters:
        - frame: the original image as a NumPy array (BGR format)
        - detections: sv.Detections object with `.bbox`, `.class_id`, `.tracker_id`
        """

        #For sanity check
        h, w, _ = frame.shape

        for bbox, class_id, track_id in zip(detections.data["original_xyxy"], detections.class_id, detections.tracker_id):
            if class_id in self.class_ids and track_id is not None:
                x_min, y_min, x_max, y_max = map(int, bbox)

                #Sanity check:------------------------

                x1, y1 = max(0, x_min), max(0, y_min)
                x2, y2 = min(w, x_max), min(h, y_max)

                if x2 <= x1 or y2 <= y1:
                    continue  # skip invalid crop

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue  # skip empty crop

                #--------------------------------------

                encoded_crop = self.prepare_crop_for_mqtt(crop)

                self.data.append({
                    'track_id': int(track_id),
                    'bbox': (x_min, y_min, x_max, y_max),
                    'encoded_crop': encoded_crop  # base64-encoded JPEG
                })

        self.send_over_mqtt()

        #self.preview_crops(wait_ms=0)  # press any key to step through, ESC to exit

    def prepare_crop_for_mqtt(self, crop):
        success, buffer = cv2.imencode('.png', crop)
        if not success:
            raise ValueError("Encoding failed.")
        return buffer.tobytes().hex()
    
    def send_over_mqtt(self):
        for entry in self.data:
            payload = {
                "track_id": entry["track_id"],
                "bbox": entry["bbox"],
                "encoded_crop": entry["encoded_crop"]
            }
            result = self.client.publish(self.mqtt_topic, json.dumps(payload))
            if result[0] != 0:
                print(f"[MQTT] Failed to send message for track_id {entry['track_id']}")

        time.sleep(0.25)  # Let messages flush
    
    def preview_crops(self, wait_ms=0):
        """
        Decodes and displays each image stored in self.data using OpenCV.

        Parameters:
        - wait_ms: Time to wait after displaying each image (0 = wait for key press)
        """
        for i, entry in enumerate(self.data):
            encoded = entry['encoded_crop']
            img_bytes = base64.b64decode(encoded)
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            crop = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if crop is None:
                print(f"[{i}] Failed to decode crop.")
                continue

            track_id = entry['track_id']
            bbox = entry['bbox']
            cv2.imshow(f'Track {track_id} BBox {bbox}', crop)
            key = cv2.waitKey(wait_ms)

            # Optional: press ESC to break early
            if key == 27:
                break

        cv2.destroyAllWindows()


    def get_data(self):
        """Returns the list of stored detection data."""
        return self.data

    def clear(self):
        """Clears the stored data (optional, if used per-frame)."""
        self.data.clear()
