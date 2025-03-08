from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import uuid
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import csv


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
DEFAULT_IMAGE_PATH = 'pic.jpeg' 
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

FACE_DISTANCE_THRESHOLD = 0.5

os.makedirs(os.path.join(UPLOAD_FOLDER, 'profile_images'), exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('reference_images', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_wearing_glasses(image, face_landmarks):
    if not face_landmarks:
        return False
    
    landmarks = face_landmarks[0]
    if not 'left_eye' in landmarks or not 'right_eye' in landmarks:
        return False
        
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    
    all_eye_points = left_eye + right_eye
    min_x = min([p[0] for p in all_eye_points]) - 10
    max_x = max([p[0] for p in all_eye_points]) + 10
    min_y = min([p[1] for p in all_eye_points]) - 10
    max_y = max([p[1] for p in all_eye_points]) + 10
    
    eye_region = image[max(0, min_y):min(image.shape[0], max_y), 
                       max(0, min_x):min(image.shape[1], max_x)]
    
    if eye_region.size == 0:
        return False
    
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_eye, 100, 200)
    
    return np.sum(edges > 0) > (eye_region.shape[0] * eye_region.shape[1] * 0.1)

def adjust_confidence(confidence, img1_has_glasses, img2_has_glasses):
    if img1_has_glasses != img2_has_glasses:
        return confidence * 0.8
    elif img1_has_glasses and img2_has_glasses:
        return confidence * 0.9 
    return confidence

def compare_faces(uploaded_image_path, reference_image_path):
    img1 = cv2.imread(uploaded_image_path)
    if img1 is None:
        return {
            'success': False,
            'message': f'Error loading uploaded image: {uploaded_image_path}'
        }
    
    img2 = cv2.imread(reference_image_path)
    if img2 is None:
        return {
            'success': False,
            'message': f'Error loading reference image: {reference_image_path}'
        }
    
    rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    face_locations1 = face_recognition.face_locations(rgb_img1)
    if not face_locations1:
        return {
            'success': False,
            'message': 'No face detected in the uploaded image.'
        }
    
    face_landmarks1 = face_recognition.face_landmarks(rgb_img1, face_locations1)
    img1_encodings = face_recognition.face_encodings(rgb_img1, face_locations1)
    if not img1_encodings:
        return {
            'success': False,
            'message': 'No face encodings could be generated for the uploaded image.'
        }
    
    img1_encoding = img1_encodings[0]
    img1_has_glasses = is_wearing_glasses(rgb_img1, face_landmarks1)
    
    face_locations2 = face_recognition.face_locations(rgb_img2)
    if not face_locations2:
        return {
            'success': False,
            'message': 'No face detected in the reference image.'
        }
    
    # Get face landmarks and encodings for second image
    face_landmarks2 = face_recognition.face_landmarks(rgb_img2, face_locations2)
    img2_encodings = face_recognition.face_encodings(rgb_img2, face_locations2)
    if not img2_encodings:
        return {
            'success': False,
            'message': 'No face encodings could be generated for the reference image.'
        }
    
    img2_encoding = img2_encodings[0]
    img2_has_glasses = is_wearing_glasses(rgb_img2, face_landmarks2)
    
    # Calculate face distance and confidence
    raw_distance = face_recognition.face_distance([img1_encoding], img2_encoding)
    raw_confidence = 1.0 - min(1.0, raw_distance[0])
    
    # Adjust confidence based on presence of glasses
    adjusted_confidence = adjust_confidence(raw_confidence, img1_has_glasses, img2_has_glasses)
    adjusted_confidence_percentage = adjusted_confidence * 100
    
    # Adjust threshold based on presence of glasses
    adjusted_threshold = FACE_DISTANCE_THRESHOLD
    if img1_has_glasses or img2_has_glasses:
        adjusted_threshold = FACE_DISTANCE_THRESHOLD * 0.9
    
    # Determine if it's a match
    is_match = raw_distance[0] <= adjusted_threshold
    
    # Determine confidence level
    confidence_level = "Very Low"
    if adjusted_confidence_percentage > 80:
        confidence_level = "Very High"
    elif adjusted_confidence_percentage > 70:
        confidence_level = "High"
    elif adjusted_confidence_percentage > 60:
        confidence_level = "Moderate"
    elif adjusted_confidence_percentage > 50:
        confidence_level = "Low"
    
    img1_with_rect = img1.copy()
    img2_with_rect = img2.copy()
    
    for i, location in enumerate(face_locations1):
        has_glasses = img1_has_glasses if i == 0 else False
        top, right, bottom, left = location
        color = (0, 255, 0) if not has_glasses else (255, 0, 0)  # Green for no glasses, red for glasses
        cv2.rectangle(img1_with_rect, (left, top), (right, bottom), color, 2)
    
    for i, location in enumerate(face_locations2):
        has_glasses = img2_has_glasses if i == 0 else False
        top, right, bottom, left = location
        color = (0, 255, 0) if not has_glasses else (255, 0, 0)  # Green for no glasses, red for glasses
        cv2.rectangle(img2_with_rect, (left, top), (right, bottom), color, 2)
    
    match_color = (0, 255, 0) if is_match else (0, 0, 255)
    cv2.putText(img1_with_rect, f"Match: {is_match}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)
    cv2.putText(img1_with_rect, f"Conf: {adjusted_confidence_percentage:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)
    cv2.putText(img1_with_rect, f"Glasses: {img1_has_glasses}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.putText(img2_with_rect, f"Match: {is_match}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)
    cv2.putText(img2_with_rect, f"Conf: {adjusted_confidence_percentage:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)
    cv2.putText(img2_with_rect, f"Glasses: {img2_has_glasses}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    h1, w1 = img1_with_rect.shape[:2]
    h2, w2 = img2_with_rect.shape[:2]
    
    target_height = min(h1, h2)
    
    aspect_ratio1 = w1 / h1
    new_width1 = int(aspect_ratio1 * target_height)
    img1_resized = cv2.resize(img1_with_rect, (new_width1, target_height))
    
    aspect_ratio2 = w2 / h2
    new_width2 = int(aspect_ratio2 * target_height)
    img2_resized = cv2.resize(img2_with_rect, (new_width2, target_height))
    
    combined_img = np.hstack((img1_resized, img2_resized))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"comparison_result_{timestamp}.jpg"
    result_path = os.path.join(RESULTS_FOLDER, result_filename)
    
    cv2.imwrite(result_path, combined_img)
    
    csv_filename = os.path.join(RESULTS_FOLDER, "face_comparison_results.csv")
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(['Image1', 'Image2', 'Glasses1', 'Glasses2', 'Match', 'Raw Distance', 
                               'Raw Confidence', 'Adjusted Confidence', 'Confidence Level', 'Timestamp'])
        
        csv_writer.writerow([
            uploaded_image_path, 
            reference_image_path,
            img1_has_glasses,
            img2_has_glasses, 
            is_match, 
            raw_distance[0], 
            f"{raw_confidence * 100:.2f}%",
            f"{adjusted_confidence_percentage:.2f}%", 
            confidence_level,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])
    
    return {
        'success': True,
        'match': bool(is_match),
        'raw_distance': float(raw_distance[0]),
        'raw_confidence': float(raw_confidence * 100),
        'adjusted_confidence': float(adjusted_confidence_percentage),
        'confidence_level': confidence_level,
        'image1_has_glasses': bool(img1_has_glasses),
        'image2_has_glasses': bool(img2_has_glasses),
        'result_image_path': result_path
    }

@app.route('/submit-identification', methods=['POST'])
def submit_identification():
    try:
        if 'profileImage' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing profile image'
            }), 400
        
        profile_image = request.files['profileImage']
        
        if profile_image.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(profile_image.filename):
            return jsonify({
                'success': False,
                'message': 'Invalid file type. Only PNG, JPG, and JPEG are allowed.'
            }), 400
        
        reference_image_path = DEFAULT_IMAGE_PATH
        if 'referenceImage' in request.files and request.files['referenceImage'].filename != '':
            reference_image = request.files['referenceImage']
            if allowed_file(reference_image.filename):
                reference_filename = f"{uuid.uuid4()}_{secure_filename(reference_image.filename)}"
                reference_path = os.path.join(app.config['UPLOAD_FOLDER'], 'reference_images', reference_filename)
                os.makedirs(os.path.dirname(reference_path), exist_ok=True)
                reference_image.save(reference_path)
                reference_image_path = reference_path
            else:
                return jsonify({
                    'success': False,
                    'message': 'Invalid reference image file type. Only PNG, JPG, and JPEG are allowed.'
                }), 400
        
        profile_filename = f"{uuid.uuid4()}_{secure_filename(profile_image.filename)}"
        profile_path = os.path.join(app.config['UPLOAD_FOLDER'], 'profile_images', profile_filename)
        profile_image.save(profile_path)
        
        comparison_result = compare_faces(profile_path, reference_image_path)
        
        if not comparison_result['success']:
            return jsonify(comparison_result), 400
        
        return jsonify({
            'success': True,
            'message': 'Face comparison completed',
            'data': comparison_result
        }), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error processing request: {str(e)}'
        }), 500

@app.route('/compare-with-multiple', methods=['POST'])
def compare_with_multiple():
    try:
        if 'profileImage' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing profile image'
            }), 400
        
        profile_image = request.files['profileImage']
        
        if profile_image.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(profile_image.filename):
            return jsonify({
                'success': False,
                'message': 'Invalid file type. Only PNG, JPG, and JPEG are allowed.'
            }), 400
        
        profile_filename = f"{uuid.uuid4()}_{secure_filename(profile_image.filename)}"
        profile_path = os.path.join(app.config['UPLOAD_FOLDER'], 'profile_images', profile_filename)
        profile_image.save(profile_path)
        
        reference_paths = []
        
        if 'referenceImages' in request.files:
            reference_files = request.files.getlist('referenceImages')
            for ref_file in reference_files:
                if ref_file.filename != '' and allowed_file(ref_file.filename):
                    ref_filename = f"{uuid.uuid4()}_{secure_filename(ref_file.filename)}"
                    ref_path = os.path.join(app.config['UPLOAD_FOLDER'], 'reference_images', ref_filename)
                    os.makedirs(os.path.dirname(ref_path), exist_ok=True)
                    ref_file.save(ref_path)
                    reference_paths.append(ref_path)
        
        if not reference_paths:
            reference_paths.append(DEFAULT_IMAGE_PATH)
        
        comparison_results = []
        best_match = None
        best_confidence = -1
        
        for ref_path in reference_paths:
            result = compare_faces(profile_path, ref_path)
            if result['success']:
                comparison_results.append(result)
                
                if result['adjusted_confidence'] > best_confidence:
                    best_confidence = result['adjusted_confidence']
                    best_match = result
        
        if not comparison_results:
            return jsonify({
                'success': False,
                'message': 'No successful comparisons could be made'
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'Face comparisons completed',
            'data': {
                'all_comparisons': comparison_results,
                'best_match': best_match
            }
        }), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error processing request: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)