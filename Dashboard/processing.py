from ultralytics import YOLO
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import math
from io import BytesIO
from PIL import Image

def calculate_distance(x1, y1, x2, y2):
    # Apply the Euclidean distance formula
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance
    

def draw_boxes(image, boxes, class_ids, confidences, class_map, thickness=1, padding=2, font_scale=0.2):
    for box, class_id, conf in zip(boxes, class_ids, confidences):
        xmin, ymin, xmax, ymax = map(int, box)
        label = f"{class_map[int(class_id)]}: {conf:.2f}"

        # Draw bounding box with custom thickness
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=thickness)

        # Calculate label size with the reduced font scale
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, 1)
        
        # Adjust the position for the label box and text
        label_ymin = max(ymin, label_size[1] + 10)

        # Draw a smaller filled rectangle for the text background (adjust height with padding)
        cv2.rectangle(
            image, 
            (xmin, label_ymin - label_size[1] - padding), 
            (xmin + label_size[0] + padding, label_ymin + padding), 
            color=(0, 0, 255), 
            thickness=-1  # Filled box
        )

        # Put the label text with reduced font size and anti-aliasing
        cv2.putText(
            image, 
            label, 
            (xmin + padding // 2, label_ymin - padding // 2),  # Add small offset for padding
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale,  # Reduced font scale
            (255, 255, 255),  # Black text
            1,  # Text thickness
            cv2.LINE_AA  # Anti-aliasing for smoother text
        )

def processImage(Path_of_image, Path_of_custom_model, Path_of_PT_model, output_imgs_path):
    output_image_list = []
    # Load your custom YOLO model
    model = YOLO(Path_of_custom_model)  # Update the path if different
    model2 = YOLO(Path_of_PT_model)

    # Image path
    img_path = Path_of_image
    image = cv2.imread(img_path)
    image2 = cv2.imread(img_path)
    # Get prediction results
    results = model(img_path)
    results2 = model2(img_path)

    # Extract results
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates (xmin, ymin, xmax, ymax)
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs

    boxes_coco = results2[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates (xmin, ymin, xmax, ymax)
    confidences_coco = results2[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids_coco = results2[0].boxes.cls.cpu().numpy()  # Class IDs

    # Save results to a text file (with xmin, ymin, xmax, ymax, class_id)
    txt_output_path = './output_boxes.txt'
    txt_output_path_coco = './output_boxes_coco.txt'

    with open(txt_output_path, 'w') as f:
        for box, class_id, conf in zip(boxes, class_ids, confidences):
            f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {int(class_id)} {conf}\n")
            
    with open(txt_output_path_coco, 'w') as f:
        for box, class_id, conf in zip(boxes_coco, class_ids_coco, confidences_coco):
            f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {int(class_id)} {conf}\n")
    # Alternatively, if you want to save it in XML format (like Pascal VOC format):
    xml_output_path = './output_boxes.xml'
    root = ET.Element("annotation")

    xml_output_path_coco = './output_boxes_coco.xml'
    root_coco = ET.Element("annotation")

    items_list = [0]*10
    class_map_custom = [
        'Hardhat', 'Vest', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
        'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
    ]

    class_map_coco = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", 
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    items_list_coco = [0]*len(class_map_coco)

    for box, class_id in zip(boxes, class_ids):
        object_tag = ET.SubElement(root, "object")
        ET.SubElement(object_tag, "name").text = str(int(class_id))
        print(str(int(class_id)))
        items_list[int(class_id)]+=1
        bndbox_tag = ET.SubElement(object_tag, "bndbox")
        ET.SubElement(bndbox_tag, "xmin").text = str(int(box[0]))
        ET.SubElement(bndbox_tag, "ymin").text = str(int(box[1]))
        ET.SubElement(bndbox_tag, "xmax").text = str(int(box[2]))
        ET.SubElement(bndbox_tag, "ymax").text = str(int(box[3]))

    print('------------------')

    for box, class_id in zip(boxes_coco, class_ids_coco):
        object_tag = ET.SubElement(root_coco, "object")
        ET.SubElement(object_tag, "name").text = str(int(class_id))
        print(str(int(class_id)))
        items_list_coco[int(class_id)]+=1
        bndbox_tag = ET.SubElement(object_tag, "bndbox")
        ET.SubElement(bndbox_tag, "xmin").text = str(int(box[0]))
        ET.SubElement(bndbox_tag, "ymin").text = str(int(box[1]))
        ET.SubElement(bndbox_tag, "xmax").text = str(int(box[2]))
        ET.SubElement(bndbox_tag, "ymax").text = str(int(box[3]))

    print('------------------')
    # Write XML tree to file
    tree = ET.ElementTree(root)
    tree_coco = ET.ElementTree(root_coco)
    tree.write(xml_output_path)

    draw_boxes(image, boxes, class_ids, confidences, class_map_custom, thickness=1, padding=2, font_scale=0.2)
    draw_boxes(image2, boxes_coco, class_ids_coco, confidences_coco, class_map_coco, thickness=1, padding=2, font_scale=0.2)
    
    # add to output image list.
    output_image_list.append(image)
    output_image_list.append(image2)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    coco_dict = {}
    custom_dict = {}

    tree_coco.write(xml_output_path_coco)
    for i in range(0,len(class_map_custom)):
        if items_list[i] > 0:
            print(f'{class_map_custom[i]} : {items_list[i]}')
            custom_dict[class_map_custom[i]] = items_list[i]
            
    for i in range(0,len(class_map_coco)):
        if items_list_coco[i] > 0:
            print(f'{class_map_coco[i]} : {items_list_coco[i]}')
            coco_dict[class_map_coco[i]] = items_list_coco[i]

    print(coco_dict)
    print(custom_dict)

    people = coco_dict.get('person', 0)
    hardhats = custom_dict.get('Hardhat', 0)
    vests = custom_dict.get('Vest', 0)
    missing_hardhats = people - hardhats
    missing_vests = people - vests
    machines = coco_dict.get('car', 0) + coco_dict.get('truck', 0)

    print(f'number of people not wearing hardhats: {missing_hardhats}')
    print(f'number of people not wearing vests: {missing_vests}')
    print(f"number of machines present: {machines}")

    # Create data points for the histogram
    # Data labels and values
    categories = ['People', 'Hardhats', 'Vests', 'Missing Hardhats', 'Missing Vests', 'Machines']
    values = [people, hardhats, vests, missing_hardhats, missing_vests, machines]

    # Create a bar chart
    plt.bar(categories, values, color=['blue', 'green', 'orange', 'red', 'purple', 'gray'])

    # Adding title and labels
    plt.title('Construction Site Data Summary')
    plt.xlabel('Categories')
    plt.ylabel('Count')

    # Display the bar chart
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')  # Save the plot as a PNG in memory
    buf.seek(0)

    # Step 3: Load buffer into a NumPy array using PIL
    image_pil = Image.open(buf)  # Open image using PIL
    image_np = np.array(image_pil)  # Convert to a NumPy array

    # Step 4: Convert RGB (matplotlib default) to BGR (OpenCV format)
    image_bar = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    output_image_list.append(image_bar)
    
    plt.show()

    people_center_pts = []
    machine_center_pts = []

    for box, class_id in zip(boxes_coco, class_ids_coco):
        xmid = (float(box[0])+float(box[2]))/2
        ymid = (float(box[1])+float(box[3]))/2
        if class_id == 0:
            people_center_pts.append((xmid, ymid))
        if class_id == 2 or class_id == 7:
            machine_center_pts.append((xmid, ymid))

    print(f'people coords: {people_center_pts}')

    height, width, channels = image2.shape

    # Print the dimensions
    print(f"Width: {width}, Height: {height}, Channels: {channels}")
    fig, ax = plt.subplots()

    # Plot people center points in blue
    if people_center_pts:  # Ensure there are people points to plot
        x_coords_people, y_coords_people = zip(*people_center_pts)
        ax.scatter(x_coords_people, y_coords_people, color='blue', label='People')

    # Plot machine center points in red
    if machine_center_pts:  # Ensure there are machine points to plot
        x_coords_machines, y_coords_machines = zip(*machine_center_pts)
        ax.scatter(x_coords_machines, y_coords_machines, color='red', label='Machines')

    # Combine people and machine center points into a single list
    all_center_pts = people_center_pts + machine_center_pts

    # Draw lines between any two entities whose distance is less than the threshold
    threshold = 300
    for i, pt1 in enumerate(all_center_pts):
        for j, pt2 in enumerate(all_center_pts):
            if i != j and calculate_distance(pt1[0], pt1[1], pt2[0], pt2[1]) < threshold:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='green', linestyle='--', linewidth=1)


    # Set limits based on rectangle size
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # Draw the rectangle outline
    rect = plt.Rectangle((0, 0), width, height, fill=False, color='black', linewidth=2)
    ax.add_patch(rect)

    # Add labels and title
    ax.set_title('Entities Center Points with Connecting Lines')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Show grid and legend
    ax.grid(True)
    ax.legend()


    # Display the plot
    plt.gca().set_aspect('equal', adjustable='box')
    
    buf = BytesIO()
    plt.savefig(buf, format='png')  # Save the plot as a PNG in memory
    buf.seek(0)

    # Step 3: Load buffer into a NumPy array using PIL
    image_pil = Image.open(buf)  # Open image using PIL
    image_np = np.array(image_pil)  # Convert to a NumPy array

    # Step 4: Convert RGB (matplotlib default) to BGR (OpenCV format)
    image_graph = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    output_image_list.append(image_graph)
    
    plt.show()
    for i in range(0, len(output_image_list)):
        cv2.imwrite(f'{output_imgs_path}output_image_{i}.png', output_image_list[i])


def main():
    img_path = './YOLOmodel/3.png'
    cus_model_path = './YOLOmodel/best.pt'
    pt_model_path = './YOLOmodel/PT_model.pt'
    output_path = '../static/'
    processImage(img_path, cus_model_path, pt_model_path, output_path)

if __name__ == '__main__':
    main()