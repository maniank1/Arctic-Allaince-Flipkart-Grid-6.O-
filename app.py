from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
import cv2
import pytesseract
from PIL import Image
import base64
import numpy as np
import threading
import time
import os

# Initialize Flask app and SocketIO
app = Flask(_name_, template_folder=os.path.abspath('templates'))
socketio = SocketIO(app, cors_allowed_origins="*")

# OCR configuration (using Tesseract)
pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'  # Adjust the path to your Tesseract executable

analyzing = False
capture_thread = None

def analyze_image(image, option):
    try:
        # Convert the PIL image to OpenCV format
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if option == 'brand':
            # Use OCR to detect text for brand recognition
            text = pytesseract.image_to_string(open_cv_image)
            if text:
                return f"Brand: {text.strip()}"
            else:
                return "Brand: None detected"

        elif option == 'expiry_mrp':
            # Use OCR to detect MRP and Expiry Date
            text = pytesseract.image_to_string(open_cv_image)
            mrp = "MRP: Not found"
            expiry_date = "Date: Not found"

            # Simple extraction logic for demo purposes
            if "MRP" in text:
                mrp = f"MRP: {text.split('MRP')[-1].strip()}"
            if "Expiry" in text or "Best Before" in text:
                expiry_date = f"Date: {text.split('Expiry')[-1].strip()}"
            
            return f"{mrp}\n{expiry_date}"

        elif option == 'auto_capture':
            # Simple OCR for extracting brand details, pack size, and brand size
            text = pytesseract.image_to_string(open_cv_image)
            return f"Extracted details: {text.strip() if text else 'Not found'}"

        elif option == 'fruit_freshness':
            # Dummy logic for fruit freshness detection (no AI)
            fruit_detected = "Yes" if "fruit" in text or "vegetable" in text else "No"
            freshness = "80%"  # Example, manually set
            return f"Fruit or Vegetable Detected: {fruit_detected}\nFreshness: {freshness}"
        
    except Exception as e:
        return f"An error occurred during image analysis: {str(e)}"

def continuous_analysis():
    global analyzing
    while analyzing:
        socketio.emit('request_frame')
        time.sleep(5)  # Analyze every 5 seconds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/brand_recognition')
def brand_recognition():
    return render_template('brand_recognition.html')

@app.route('/expiry_mrp')
def expiry_mrp():
    return render_template('expiry_and_mrp.html')

@app.route('/branddetails_and_packsize')
def branddetails_and_packsize():
    return render_template('branddetails_and_packsize.html')

@app.route('/fruit_freshness')
def fruit_freshness():
    return render_template('freshness_detetction.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.template_folder, filename)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    global analyzing
    analyzing = False

@socketio.on('start_analysis')
def handle_start_analysis(data):
    global analyzing, capture_thread
    analyzing = True
    capture_thread = threading.Thread(target=continuous_analysis)
    capture_thread.daemon = True
    capture_thread.start()
    print(f"Analysis started with option: {data.get('option')}")

@socketio.on('stop_analysis')
def handle_stop_analysis():
    global analyzing
    analyzing = False
    print("Analysis stopped")

@socketio.on('frame')
def handle_frame(data):
    try:
        # Extract the base64 image data
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Analyze the image based on selected option
        analysis_result = analyze_image(pil_image, data['option'])
        
        # Send results back to client
        socketio.emit('analysis_result', {'result': analysis_result})
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        socketio.emit('analysis_result', {'error': str(e)})

if _name_ == '_main_':
    print(f"Templates directory: {app.template_folder}")
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)