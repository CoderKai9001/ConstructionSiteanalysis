import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set to evaluation mode

# Define the transformation: Convert image to Tensor
transform = transforms.Compose([
    transforms.ToTensor()  # Converts the image to a Tensor
])

# Load and preprocess the image
image_path = './csiteimgs/5.png'  # Replace with the path to your image
image = Image.open(image_path)
image_tensor = transform(image)  # Apply the transformation
image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension (1, C, H, W)

# Run the model on the image
with torch.no_grad():  # No need to compute gradients for inference
    predictions = model(image_tensor)

# Get the bounding boxes, labels, and scores from the model output
boxes = predictions[0]['boxes'].numpy()  # Bounding boxes
labels = predictions[0]['labels'].numpy()  # Class labels
scores = predictions[0]['scores'].numpy()  # Confidence scores

# Filter the results based on confidence score threshold
threshold = 0.7  # Only display detections with a confidence score higher than 0.5
filtered_boxes = boxes[scores > threshold]
filtered_labels = labels[scores > threshold]
filtered_scores = scores[scores > threshold]

# Plot the image and the detected bounding boxes
fig, ax = plt.subplots(1, figsize=(12, 9))

# Display the image
ax.imshow(image)

# Draw bounding boxes and labels on the image
for i, box in enumerate(filtered_boxes):
    # Create a rectangle patch for the bounding box
    rect = patches.Rectangle(
        (box[0], box[1]),  # Bottom left corner
        box[2] - box[0],  # Width
        box[3] - box[1],  # Height
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)
    
    class_name = COCO_INSTANCE_CATEGORY_NAMES[filtered_labels[i]]
    
    # Display label and confidence score
    label_text = f"{class_name}: {filtered_scores[i]:.2f}"
    ax.text(box[0], box[1] - 10, label_text, color='white', fontsize=12, backgroundcolor='red')

# Show the image with bounding boxes
plt.show()
