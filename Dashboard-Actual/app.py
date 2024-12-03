import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, render_template , request , jsonify 
from time import sleep
import json
import os

#Creating flask app
app = Flask(__name__)

# Path to the JSON file containing the regions data
DATA_FILE = 'regions.json'
# Path to your service account key file
service_account_path = "csite-assistant-firebase-adminsdk-p20ha-3f313d174f.json"

# Initialize the Firebase Admin SDK

cred = credentials.Certificate(service_account_path)
firebase_admin.initialize_app(cred , {
    'databaseURL':"https://csite-assistant-default-rtdb.firebaseio.com/"
})

def load_regions():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def save_regions(regions):
    with open(DATA_FILE, 'w') as f:
        json.dump(regions, f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/work_assign.html')
def work_assign():
    return render_template('work_assign.html')

@app.route('/add_region', methods=['POST'])
def add_region():
    data = request.get_json()
    regions = load_regions()
    regions.append(data)
    save_regions(regions)
    return jsonify({'success': True})

@app.route('/get_regions')
def get_regions():
    regions = load_regions()
    return jsonify(regions)

@app.route('/api/object_detections')
def get_object_detections():
    object_detections = db.reference('/').get()
    print("Object Detections Data:", object_detections)  # Debug Statement

    if not object_detections or 'object_detections' not in object_detections:
        return jsonify({'labels': [], 'counts': []}), 200

    detections = object_detections['object_detections']
    label_counts = {}

    for detection in detections:
        if isinstance(detection, dict):
            label = detection.get('label', 'Unknown')
            label_counts[label] = label_counts.get(label, 0) + 1
        else:
            print(f"Invalid detection format: {detection}")

    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    return jsonify({'labels': labels, 'counts': counts}), 200

# @app.route('/api/bounding_box_centers')
# def get_bounding_box_centers():
#     object_detections = db.reference('/').get()
#     print("Object Detections Data:", object_detections)  # Debug Statement
#     centers = []

#     if object_detections and 'object_detections' in object_detections:
#         detections = object_detections['object_detections']
#         for detection in detections:
#             bounding_box = detection.get('boundingBox', {})
#             top = bounding_box.get('top', 0)
#             bottom = bounding_box.get('bottom', 0)
#             left = bounding_box.get('left', 0)
#             right = bounding_box.get('right', 0)
            
#             center_x = (left + right) / 2
#             center_y = (top + bottom) / 2
#             centers.append({'x': center_x, 'y': center_y})
#     else:
#         print("No object detections found or incorrect data structure.")

#     return jsonify({'centers': centers}), 200
@app.route('/api/bounding_box_centers')
def get_bounding_box_centers():
    object_detections = db.reference('/').get()
    print("Object Detections Data:", object_detections)  # Debug Statement
    centers = []

    if object_detections and 'object_detections' in object_detections:
        detections = object_detections['object_detections']
        for detection in detections:
            if isinstance(detection, dict):
                bbox = detection.get('bounding_box')
                if bbox:
                    center_x = bbox.get('x') + bbox.get('width') / 2
                    center_y = bbox.get('y') + bbox.get('height') / 2
                    centers.append({'x': center_x, 'y': center_y})
    return jsonify({'centers': centers}), 200

if __name__ == '__main__':
    app.run(debug=True)