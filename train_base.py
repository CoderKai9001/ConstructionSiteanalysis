# from ultralytics import YOLO
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained YOLOv8 model
model = YOLO('yolov10x.pt')
# model = YOLO('yolov8l.pt')
# model = YOLO('yolov8x.pt')
# You can use 'yolov8s.pt', 'yolov8m.pt', etc., depending on your needs.

# Load the imageScreenshot 2024-09-17 at 3.55.28 PM.png
image_path = "./csiteimgs/4.png"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image at {image_path}. Please check the file path.")
else:
    # Display the image using Matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color rendering
    plt.title('Loaded Image')
    plt.axis('off')  # Turn off axis labels
    plt.show()

# Perform object detection
results = model(image , conf=0.25)

# Results contain information about the detected objects
print(results)  # You can inspect results and extract useful information

# Visualize the detections on the image
annotated_image = results[0].plot()

# Display the image with detections
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()