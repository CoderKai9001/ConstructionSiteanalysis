# Print the data Drawing the bounding boxes
# if(object_detections is None):
#     print("No data in the database")
# else:
#     plt.figure(figsize=(10, 8))
#     ax = plt.gca()

#     for detection in object_detections:
#         label = detection.get('label')
#         confidence = detection.get('confidence')
#         bounding_box = detection.get('boundingBox', {})
        
#         top = bounding_box.get('top')
#         bottom = bounding_box.get('bottom')
#         left = bounding_box.get('left')
#         right = bounding_box.get('right')
        
#         print(f"Label: {label}")
#         print(f"Confidence: {confidence}")
#         print(f"Bounding Box - Top: {top}, Bottom: {bottom}, Left: {left}, Right: {right}\n")
        
#         # Calculate width and height
#         width = right - left
#         height = bottom - top
        
#         # Create a Rectangle patch
#         rect = Rectangle((left, top), width, height, linewidth=2, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
#         plt.text(left, top, f'{label} ({confidence:.2f})', color='yellow', fontsize=12, backgroundcolor='red')

#     # Set limits based on bounding boxes
#     all_left = [d['boundingBox']['left'] for d in object_detections]
#     all_top = [d['boundingBox']['top'] for d in object_detections]
#     all_right = [d['boundingBox']['right'] for d in object_detections]
#     all_bottom = [d['boundingBox']['bottom'] for d in object_detections]
    
#     plt.xlim(min(all_left) - 10, max(all_right) + 10)
#     plt.ylim(max(all_bottom) + 10, min(all_top) - 10)
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.title('Object Detections Bounding Boxes')
#     plt.show()

# ref = db.reference('/')  # Root of the database
# data = ref.get()  # Get all data as JSON
# print("Data from Firebase:")
# print(data)
# # Accessing data from Firebase Realtime Database
# object_detections = db.reference('/').get()



# Printing the data and the bar chart
# Print the data
# if(object_detections is None):
#     print("No data in the database")
# else:
#     # Count occurrences of each label
#     label_counts = {}
#     for detection in object_detections:
#         label = detection.get('label', 'Unknown')
#         label_counts[label] = label_counts.get(label, 0) + 1
    
#     # Create a bar graph
#     plt.figure(figsize=(10, 8))
#     labels = list(label_counts.keys())
#     counts = list(label_counts.values())
    
#     plt.bar(labels, counts, color='skyblue')
#     plt.xlabel('Object Types')
#     plt.ylabel('Count')
#     plt.title('Types of Objects Detected')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()