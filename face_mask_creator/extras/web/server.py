"""Web server implementation for face-mask-creator."""

import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from face_mask_creator import FaceMaskCreator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the face mask creator
creator = FaceMaskCreator()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/create_mask', methods=['POST'])
def create_mask():
    """Create a mask from an uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'mask_{filename}')
        
        # Save uploaded file
        file.save(input_path)
        
        try:
            # Create mask
            mask = creator.create_mask(input_path)
            mask.save(output_path)
            
            # Return the mask image
            return send_file(output_path, mimetype='image/png')
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up uploaded files
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)

def start_server(host='0.0.0.0', port=5000, debug=False):
    """Start the web server."""
    app.run(host=host, port=port, debug=debug) 