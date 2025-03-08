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
import cx_Oracle
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
PROFILE_FOLDER = os.path.join(UPLOAD_FOLDER, 'profiles')
AADHAR_FOLDER = os.path.join(UPLOAD_FOLDER, 'aadhars')
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directories exist
os.makedirs(PROFILE_FOLDER, exist_ok=True)
os.makedirs(AADHAR_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('reference_images', exist_ok=True)

# Oracle DB Connection settings
DB_CONFIG = {
    "user": "hr",
    "password": "hr",
    "connectionString": "localhost/xepdb1"
}

# Face comparison settings
FACE_DISTANCE_THRESHOLD = 0.5

def get_db_connection():
    """Create and return a connection to the Oracle database"""
    connection = cx_Oracle.connect(
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        dsn=DB_CONFIG["connectionString"]
    )
    return connection

def save_base64_image(base64_data, folder, filename_prefix):
    """Save a base64 encoded image to a file"""
    if ',' in base64_data:
        base64_data = base64_data.split(',', 1)[1]
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.jpg"
    filepath = os.path.join(folder, secure_filename(filename))
    
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(base64_data))
    
    return filepath

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
    
    face_landmarks2 = face_recognition.face_landmarks(rgb_img2, face_locations2)
    img2_encodings = face_recognition.face_encodings(rgb_img2, face_locations2)
    if not img2_encodings:
        return {
            'success': False,
            'message': 'No face encodings could be generated for the reference image.'
        }
    
    img2_encoding = img2_encodings[0]
    img2_has_glasses = is_wearing_glasses(rgb_img2, face_landmarks2)
    
    raw_distance = face_recognition.face_distance([img1_encoding], img2_encoding)
    raw_confidence = 1.0 - min(1.0, raw_distance[0])
    
    adjusted_confidence = adjust_confidence(raw_confidence, img1_has_glasses, img2_has_glasses)
    adjusted_confidence_percentage = adjusted_confidence * 100
    
    adjusted_threshold = FACE_DISTANCE_THRESHOLD
    if img1_has_glasses or img2_has_glasses:
        adjusted_threshold = FACE_DISTANCE_THRESHOLD * 0.9
    
    is_match = raw_distance[0] <= adjusted_threshold
    
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
        color = (0, 255, 0) if not has_glasses else (255, 0, 0)
        cv2.rectangle(img1_with_rect, (left, top), (right, bottom), color, 2)
    
    for i, location in enumerate(face_locations2):
        has_glasses = img2_has_glasses if i == 0 else False
        top, right, bottom, left = location
        color = (0, 255, 0) if not has_glasses else (255, 0, 0)
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
        
        # Save the uploaded image temporarily
        profile_filename = f"{uuid.uuid4()}_{secure_filename(profile_image.filename)}"
        profile_path = os.path.join(app.config['UPLOAD_FOLDER'], 'profile_images', profile_filename)
        profile_image.save(profile_path)
        
        # Fetch all profile images from the database
        connection = get_db_connection()
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT id, cpf_number, profile_image_path 
            FROM identification_data
        """)
        
        profile_records = cursor.fetchall()
        cursor.close()
        connection.close()
        
        if not profile_records:
            return jsonify({
                'success': False,
                'message': 'No profile images found in the database.'
            }), 404
        
        # Compare the uploaded image with each profile image in the database
        for record in profile_records:
            reference_image_path = record[2]  # profile_image_path from the database
            
            # Debug: Log the reference image path
            print(f"Comparing with reference image: {reference_image_path}")
            
            # Compare the uploaded image with the current reference image
            comparison_result = compare_faces(profile_path, reference_image_path)
            
            # Debug: Log the comparison result
            print(f"Comparison result: {comparison_result}")
            
            # If a match is found, return immediately
            if comparison_result['success'] and comparison_result['match']:
                # Clean up the temporary uploaded image
                os.remove(profile_path)
                
                return jsonify({
                    'success': True,
                    'message': 'Verified user exists.',
                    'data': {
                        'id': record[0],  # User ID from the database
                        'cpf_number': record[1],  # CPF number from the database
                        'confidence': comparison_result['adjusted_confidence'],
                        'result_image_path': comparison_result['result_image_path']
                    }
                }), 200
        
        # If no match is found after all comparisons, return "No verified user found"
        os.remove(profile_path)
        return jsonify({
            'success': False,
            'message': 'No verified user found.'
        }), 404
    
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
            reference_paths.append('pic.jpeg')
        
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

@app.route('/api/submit-identification', methods=['POST'])
def api_submit_identification():
    try:
        data = request.json
        cpf_number = data.get('cpfNumber')
        profile_image_base64 = data.get('profileImage')
        aadhar_image_base64 = data.get('aadharImage')
        
        if not cpf_number or len(cpf_number) != 5:
            return jsonify({"error": "CPF number must be exactly 5 characters"}), 400
        
        if not profile_image_base64:
            return jsonify({"error": "Profile image is required"}), 400
            
        if not aadhar_image_base64:
            return jsonify({"error": "Aadhar image is required"}), 400
        
        profile_path = save_base64_image(
            profile_image_base64, 
            PROFILE_FOLDER, 
            f"profile_{cpf_number}"
        )
        
        aadhar_path = save_base64_image(
            aadhar_image_base64, 
            AADHAR_FOLDER, 
            f"aadhar_{cpf_number}"
        )
        
        connection = get_db_connection()
        cursor = connection.cursor()
        
        sql = """
        INSERT INTO identification_data 
        (cpf_number, profile_image_path, aadhar_image_path) 
        VALUES (:cpf_number, :profile_path, :aadhar_path)
        """
        
        cursor.execute(sql, {
            'cpf_number': cpf_number,
            'profile_path': profile_path,
            'aadhar_path': aadhar_path
        })
        
        connection.commit()
        
        cursor.execute("SELECT id FROM identification_data WHERE cpf_number = :cpf_number ORDER BY submission_date DESC", 
                      {'cpf_number': cpf_number})
        record_id = cursor.fetchone()[0]
        
        cursor.close()
        connection.close()
        
        return jsonify({
            "success": True,
            "message": "Identification submitted successfully",
            "id": record_id
        })
        
    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500

@app.route('/api/check-status/<cpf_number>', methods=['GET'])
def check_status(cpf_number):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT id, status, submission_date 
            FROM identification_data 
            WHERE cpf_number = :cpf_number
            ORDER BY submission_date DESC
        """, {'cpf_number': cpf_number})
        
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if not result:
            return jsonify({"error": "No record found for the given CPF number"}), 404
            
        return jsonify({
            "id": result[0],
            "status": result[1],
            "submission_date": result[2].strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500

@app.route('/api/get-profile-photo/<cpf_number>', methods=['GET'])
def get_profile_photo(cpf_number):
    try:
        if not cpf_number or len(cpf_number) != 5:
            return jsonify({"error": "Invalid CPF number format"}), 400
            
        connection = get_db_connection()
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT profile_image_path 
            FROM identification_data 
            WHERE cpf_number = :cpf_number
            ORDER BY submission_date DESC
        """, {'cpf_number': cpf_number})
        
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if not result or not result[0]:
            return jsonify({"error": "No profile photo found for the given CPF number"}), 404
            
        image_path = result[0]
        
        if not os.path.exists(image_path):
            return jsonify({"error": "Profile photo file not found"}), 404
            
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
        return jsonify({
            "success": True,
            "cpf_number": cpf_number,
            "image_data": image_data,
            "content_type": "image/jpeg"
        })
        
    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500

@app.route('/api/get-aadhar-photo/<cpf_number>', methods=['GET'])
def get_aadhar_photo(cpf_number):
    try:
        if not cpf_number or len(cpf_number) != 5:
            return jsonify({"error": "Invalid CPF number format"}), 400
            
        connection = get_db_connection()
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT aadhar_image_path 
            FROM identification_data 
            WHERE cpf_number = :cpf_number
            ORDER BY submission_date DESC
        """, {'cpf_number': cpf_number})
        
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if not result or not result[0]:
            return jsonify({"error": "No Aadhar photo found for the given CPF number"}), 404
            
        image_path = result[0]
        
        if not os.path.exists(image_path):
            return jsonify({"error": "Aadhar photo file not found"}), 404
            
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
        return jsonify({
            "success": True,
            "cpf_number": cpf_number,
            "image_data": image_data,
            "content_type": "image/jpeg"
        })
        
    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500

@app.route('/api/get-user-data/<cpf_number>', methods=['GET'])
def get_user_data(cpf_number):
    try:
        if not cpf_number or len(cpf_number) != 5:
            return jsonify({"error": "Invalid CPF number format"}), 400
            
        connection = get_db_connection()
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT id, cpf_number, profile_image_path, aadhar_image_path, 
                   submission_date, status
            FROM identification_data 
            WHERE cpf_number = :cpf_number
            ORDER BY submission_date DESC
        """, {'cpf_number': cpf_number})
        
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if not result:
            return jsonify({"error": "No data found for the given CPF number"}), 404
            
        user_data = {
            "id": result[0],
            "cpf_number": result[1],
            "submission_date": result[4].strftime("%Y-%m-%d %H:%M:%S"),
            "status": result[5]
        }
        
        profile_path = result[2]
        if profile_path and os.path.exists(profile_path):
            with open(profile_path, "rb") as image_file:
                user_data["profile_image"] = base64.b64encode(image_file.read()).decode('utf-8')
        
        aadhar_path = result[3]
        if aadhar_path and os.path.exists(aadhar_path):
            with open(aadhar_path, "rb") as image_file:
                user_data["aadhar_image"] = base64.b64encode(image_file.read()).decode('utf-8')
            
        return jsonify({
            "success": True,
            "user_data": user_data
        })
        
    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)