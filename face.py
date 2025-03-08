import face_recognition
import cv2
import csv
import numpy as np
import os
from datetime import datetime

FACE_DISTANCE_THRESHOLD = 0.5 

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
    
    # Extract eye region
    eye_region = image[max(0, min_y):min(image.shape[0], max_y), 
                       max(0, min_x):min(image.shape[1], max_x)]
    
    if eye_region.size == 0:
        return False
    
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_RGB2GRAY)
    
    edges = cv2.Canny(gray_eye, 100, 200)
    
    return np.sum(edges > 0) > (eye_region.shape[0] * eye_region.shape[1] * 0.1)

def adjust_confidence(confidence, img1_has_glasses, img2_has_glasses):
    if img1_has_glasses != img2_has_glasses:
        return confidence * 0.8  # Reduce confidence by 20%
    elif img1_has_glasses and img2_has_glasses:
        return confidence * 0.9  # Reduce confidence by 10%
    return confidence

img_path = 'pic4.jpeg'
img = cv2.imread(img_path)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

face_locations = face_recognition.face_locations(rgb_img)
if not face_locations:
    print("No face detected in the first image.")
    exit()

face_landmarks = face_recognition.face_landmarks(rgb_img, face_locations)
img_encodings = face_recognition.face_encodings(rgb_img, face_locations)
if not img_encodings:
    print("No face encodings could be generated for the first image.")
    exit()
    
img_encodings = img_encodings[0]  
img1_has_glasses = is_wearing_glasses(rgb_img, face_landmarks)

img2_path = 'pic3.jpeg'
img2 = cv2.imread(img2_path)
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

face_locations2 = face_recognition.face_locations(rgb_img2)
if not face_locations2:
    print("No face detected in the second image.")
    exit()

face_landmarks2 = face_recognition.face_landmarks(rgb_img2, face_locations2)
img2_encodings = face_recognition.face_encodings(rgb_img2, face_locations2)
if not img2_encodings:
    print("No face encodings could be generated for the second image.")
    exit()
    
img2_encodings = img2_encodings[0] 
img2_has_glasses = is_wearing_glasses(rgb_img2, face_landmarks2)

raw_distance = face_recognition.face_distance([img_encodings], img2_encodings)
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

with open('face_comparison_results.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Image1', 'Image2', 'Glasses1', 'Glasses2', 'Match', 'Raw Distance', 
                         'Raw Confidence', 'Adjusted Confidence', 'Confidence Level', 'Timestamp'])
    csv_writer.writerow([
        img_path, 
        img2_path,
        img1_has_glasses,
        img2_has_glasses, 
        is_match, 
        raw_distance[0], 
        f"{raw_confidence * 100:.2f}%",
        f"{adjusted_confidence_percentage:.2f}%", 
        confidence_level,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ])

print(f"Match: {is_match}")
print(f"Image 1 has glasses: {img1_has_glasses}")
print(f"Image 2 has glasses: {img2_has_glasses}")
print(f"Raw Distance: {raw_distance[0]} (lower is more similar)")
print(f"Raw Confidence: {raw_confidence * 100:.2f}%")
print(f"Adjusted Confidence: {adjusted_confidence_percentage:.2f}%")
print(f"Confidence Level: {confidence_level}")

def draw_face_rectangle(image, location, has_glasses):
    top, right, bottom, left = location
    color = (0, 255, 0) if not has_glasses else (255, 0, 0)  # Green for no glasses, red for glasses
    cv2.rectangle(image, (left, top), (right, bottom), color, 2)
    return image

img_with_rect = img.copy()
img2_with_rect = img2.copy()

for i, location in enumerate(face_locations):
    has_glasses = img1_has_glasses if i == 0 else False
    img_with_rect = draw_face_rectangle(img_with_rect, location, has_glasses)
    
for i, location in enumerate(face_locations2):
    has_glasses = img2_has_glasses if i == 0 else False
    img2_with_rect = draw_face_rectangle(img2_with_rect, location, has_glasses)

match_color = (0, 255, 0) if is_match else (0, 0, 255)  
cv2.putText(img_with_rect, f"Match: {is_match}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)
cv2.putText(img_with_rect, f"Conf: {adjusted_confidence_percentage:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)
cv2.putText(img_with_rect, f"Glasses: {img1_has_glasses}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

cv2.putText(img2_with_rect, f"Match: {is_match}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)
cv2.putText(img2_with_rect, f"Conf: {adjusted_confidence_percentage:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)
cv2.putText(img2_with_rect, f"Glasses: {img2_has_glasses}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

h1, w1 = img_with_rect.shape[:2]
h2, w2 = img2_with_rect.shape[:2]

target_height = min(h1, h2)

aspect_ratio1 = w1 / h1
new_width1 = int(aspect_ratio1 * target_height)
img_resized = cv2.resize(img_with_rect, (new_width1, target_height))

aspect_ratio2 = w2 / h2
new_width2 = int(aspect_ratio2 * target_height)
img2_resized = cv2.resize(img2_with_rect, (new_width2, target_height))

combined_img = np.hstack((img_resized, img2_resized))

cv2.imshow('Face Comparison', combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()