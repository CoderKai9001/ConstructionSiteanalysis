import torch
from ultralytics import YOLO

# Load the YOLO model (YOLOv5 or YOLOv8)
model = YOLO('yolov10n.pt')  # Use yolov5m.pt, yolov5l.pt, etc. for different model sizes

# Train the model
model.train(
    data='data2.yaml',         # Path to the dataset YAML file
    imgsz=640,                # Image size (can be 640, 512, etc.)
    epochs=2,                # Number of epochs
    batch=4,            # Batch size (adjust based on available memory)
    device='cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
    workers=2,                # Number of workers for data loading
    project='yolo_training',  # Directory to save results
    name='custom_yolo'        # Name for the experiment
)
