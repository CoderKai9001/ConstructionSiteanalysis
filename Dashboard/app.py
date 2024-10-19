from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import processing  # Assuming this contains your processImage function

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'static/processed/'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run your YOLO model and CV processing
        img_path = filepath
        cus_model_path = './YOLOmodel/best.pt'
        pt_model_path = './YOLOmodel/PT_model.pt'
        output_folder = app.config['PROCESSED_FOLDER']
        
        # Call the processImage function to generate and save 4 images
        processing.processImage(img_path, cus_model_path, pt_model_path, output_folder)

        # After processing, redirect to the result page to show the images
        return redirect(url_for('show_result'))

@app.route('/result')
def show_result():
    return render_template('result.html', 
                           output_image1='processed/output_image_0.png',
                           output_image2='processed/output_image_1.png',
                           output_image3='processed/output_image_2.png',
                           output_image4='processed/output_image_3.png')

@app.route('/static/processed/<filename>')
def display_processed_image(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
