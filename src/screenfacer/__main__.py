import datetime
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import ImageGrab
import numpy as np
import cv2


def main():
# Capture screenshot
    screenshot = ImageGrab.grab()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    height, width = frame.shape[:2]

# Download and load model
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    model = YOLO(model_path)

# Run detection
    results = model(frame)
    detections = Detections.from_ultralytics(results[0])

# Extract and display each detected face
    # Extract and display each detected face
    if len(detections.xyxy) > 0:
        for i, bbox in enumerate(detections.xyxy):
            # Expand bbox by 50px while keeping within image bounds
            x1 = max(0, int(bbox[0]) - 50)
            y1 = max(0, int(bbox[1]) - 50)
            x2 = min(width, int(bbox[2]) + 50)
            y2 = min(height, int(bbox[3]) + 50)
            
            face = frame[y1:y2, x1:x2]
            
            # Display each face in a separate window
            cv2.imshow(f"Face {i+1}", face)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
