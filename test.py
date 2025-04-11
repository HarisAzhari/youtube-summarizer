# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import os
import uuid
import io

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3001"}})  # Allow requests from localhost:3000

# Configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size


@app.route('/api/convert', methods=['POST'])
def convert_png_to_jpg():
    """Convert a PNG image to JPG format."""
    # Check if file is in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    
    file = request.files['image']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file type
    if not file.filename.lower().endswith('.png'):
        return jsonify({'error': 'Only PNG files are accepted'}), 400
    
    try:
        # Read the image
        img_data = file.read()
        img = Image.open(io.BytesIO(img_data))
        
        # Handle transparency for PNG
        if img.mode in ('RGBA', 'LA'):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save as JPG
        img.save(filepath, "JPEG", quality=90)
        
        # Return the URL to the converted image
        return jsonify({
            'success': True,
            'jpgUrl': f"/api/images/{filename}"
        })
    
    except Exception as e:
        return jsonify({'error': f'Image conversion failed: {str(e)}'}), 500


@app.route('/api/images/<filename>')
def get_image(filename):
    """Serve the converted image."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)