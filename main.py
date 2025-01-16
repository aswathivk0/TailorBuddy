# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np
import math
import os
import sys
import time

class UserInfo:
    def __init__(self, age, gender, height_cm,camera_fov_deg):
        self.age = age
        self.gender = gender
        self.height_cm = height_cm
        #self.weight_kg = weight_kg
        self.camera_fov_deg = camera_fov_deg

def get_user_info():
    age = int(input("Enter your age: "))
    gender = input("Enter your gender: ")
    height_cm = float(input("Enter your height in cm: "))
    

    return UserInfo(age, gender, height_cm,camera_fov_deg)


class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=static_image_mode,
                                      model_complexity=model_complexity,
                                      enable_segmentation=enable_segmentation,
                                      min_detection_confidence=min_detection_confidence)
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_landmarks(self, image):
        # Convert the BGR image to RGB before processing
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        return results.pose_landmarks

    def draw_landmarks(self, image, landmarks):
        self.mp_drawing.draw_landmarks(image, landmarks, self.mp_pose.POSE_CONNECTIONS)

# Utility function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

# Function to get user inputs
def get_user_info():
    print("=== Virtual Body Measurement for Tailoring ===\n")
    
    # Get Age
    while True:
        try:
            age = int(input("Enter Age (years): "))
            if age <= 0:
                print("Age must be a positive integer.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer for Age.")
    
    # Get Gender
    
    while True:
        gender = input("Enter Gender (Male/Female/Other): ").strip().lower()
        if gender in ['m', 'f', 'other']:
            gender = gender.capitalize()
            break
        else:
            print("Invalid input. Please enter 'Male', 'Female', or 'Other' for Gender.")
    
    # Get Height
    while True:
        try:
            height_cm = float(input("Enter Height (cm): "))
            
            if height_cm <= 0:
                print("Height must be a positive number.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number for Height.")
            
    camera_fov_deg = 65
    # # Get Weight
    # while True:
    #     try:
    #         weight_kg = float(input("Enter Weight (kg): "))
    #         if weight_kg <= 0:
    #             print("Weight must be a positive number.")
    #             continue
    #         break
    #     except ValueError:
    #         print("Invalid input. Please enter a valid number for Weight.")
    
    # # Get Camera's Horizontal Field of View
    # while True:
    #     try:
    #         camera_fov_deg = float(input("Enter Camera's Horizontal Field of View (degrees): "))
    #         if camera_fov_deg <= 0:
    #             print("Field of View must be a positive number.")
    #             continue
    #         break
    #     except ValueError:
    #         print("Invalid input. Please enter a valid number for Field of View.")
    
    return UserInfo(age, gender, height_cm,camera_fov_deg)

# Function to extract required landmarks
def extract_landmarks(landmarks, image_shape):
    landmark_dict = {}
    h, w, _ = image_shape
    for id, lm in enumerate(landmarks.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        landmark_name = mp_pose_landmark_to_name(id)
        if landmark_name:
            landmark_dict[landmark_name] = (cx, cy)
    return landmark_dict

# Mapping of MediaPipe landmarks to names
def mp_pose_landmark_to_name(id):
    landmark_names = {
        0: "Nose",
        1: "Left Eye Inner",
        2: "Left Eye",
        3: "Left Eye Outer",
        4: "Right Eye Inner",
        5: "Right Eye",
        6: "Right Eye Outer",
        7: "Left Ear",
        8: "Right Ear",
        9: "Mouth Left",
        10: "Mouth Right",
        11: "Left Shoulder",
        12: "Right Shoulder",
        13: "Left Elbow",
        14: "Right Elbow",
        15: "Left Wrist",
        16: "Right Wrist",
        17: "Left Pinky",
        18: "Right Pinky",
        19: "Left Index",
        20: "Right Index",
        21: "Left Thumb",
        22: "Right Thumb",
        23: "Left Hip",
        24: "Right Hip",
        25: "Left Knee",
        26: "Right Knee",
        27: "Left Ankle",
        28: "Right Ankle",
        29: "Left Heel",
        30: "Right Heel",
        31: "Left Foot Index",
        32: "Right Foot Index"
    }
    return landmark_names.get(id, None)

# Function to calculate focal length in pixels

def calculate_focal_length_pixels(image_width, camera_fov_deg):
    # Convert FOV from degrees to radians
    camera_fov_rad = math.radians(camera_fov_deg)
    # Calculate focal length in pixels
    focal_length_px = (image_width / 2) / math.tan(camera_fov_rad / 2)
    return focal_length_px

# Function to calculate measurements using optical formulas
def calculate_measurements(front_landmarks, side_landmarks, user_info, focal_length_px):
    measurements = {}
    
    # Find ankle and shoulder in front view
    ankle = front_landmarks.get("Left Ankle") or front_landmarks.get("Right Ankle")
    shoulder = front_landmarks.get("Left Shoulder") or front_landmarks.get("Right Shoulder")
    
    if not ankle or not shoulder:
        print("Cannot find ankle or shoulder in front image for scaling.")
        return measurements
    
    # Calculate the pixel height of the user in the image
    pixel_height = ankle[1] - shoulder[1]
    if pixel_height <= 0:
        print("Invalid pixel height measurement.")
        return measurements
    
    # Calculate the distance to the camera using the user's height
    distance_cm = (focal_length_px * user_info.height_cm) / pixel_height
    print(f"Estimated Distance to Camera: {distance_cm:.2f} cm")
    
    # Calculate cm per pixel scaling factor
    cm_per_pixel = user_info.height_cm / pixel_height
    print(f"Scaling factor: {cm_per_pixel:.2f} cm per pixel")
    

    
    left_shoulder = front_landmarks.get("Left Shoulder")
    right_shoulder = front_landmarks.get("Right Shoulder")
    if left_shoulder and right_shoulder:
        shoulder_width_pixels = calculate_distance(left_shoulder, right_shoulder)
        neck_circumference_cm = shoulder_width_pixels * cm_per_pixel * 0.9
        measurements['Neck Circumference (cm)'] = round(neck_circumference_cm, 2)
    else:
        measurements['Neck Circumference (cm)'] = "N/A"
    
    # Chest circumference
    if left_shoulder and right_shoulder:
        chest_circumference_cm = shoulder_width_pixels * cm_per_pixel * 1.95  # Adjusted factor for accuracy
        measurements['Chest Circumference (cm)'] = round(chest_circumference_cm, 2)
    else:
        measurements['Chest Circumference (cm)'] = "N/A"
    
    # Waist circumference
    left_hip = front_landmarks.get("Left Hip")
    right_hip = front_landmarks.get("Right Hip")
    if left_hip and right_hip:
        waist_width_pixels = calculate_distance(left_hip, right_hip)
        waist_circumference_cm = waist_width_pixels * cm_per_pixel * 2.7  # Adjusted factor for accuracy
        measurements['Waist Circumference (cm)'] = round(waist_circumference_cm, 2)
    else:
        measurements['Waist Circumference (cm)'] = "N/A"
    
    # Wrist circumference
    left_wrist = front_landmarks.get("Left Wrist")
    right_wrist = front_landmarks.get("Right Wrist")
    if left_wrist and right_wrist:
        wrist_width_pixels = calculate_distance(left_wrist, right_wrist)
        wrist_circumference_cm = wrist_width_pixels * cm_per_pixel * 0.3 # Adjusted factor for accuracy
        measurements['Wrist Circumference (cm)'] = round(wrist_circumference_cm, 2)
    else:
        measurements['Wrist Circumference (cm)'] = "N/A"
    
  # Arm Length Measurement
    left_shoulder = front_landmarks.get("Left Shoulder")
    left_wrist = front_landmarks.get("Left Wrist")

    if left_shoulder and left_wrist:
        arm_length_pixels = calculate_distance(left_shoulder, left_wrist)
        cm_per_pixel = 0.5  # This should be determined based on your calibration
        arm_length_cm = arm_length_pixels * cm_per_pixel  # Adjust as needed for accuracy
        measurements['Arm Length (cm)'] = round(arm_length_cm, 2)
    else:
        measurements['Arm Length (cm)'] = "N/A"


    
    # Shoulder circumference approximation
    if left_shoulder and right_shoulder:
        shoulder_width_pixels = calculate_distance(left_shoulder, right_shoulder)
        neck_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        neck_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        neck_center = (neck_center_x, neck_center_y)
        shoulder_circumference_cm = shoulder_width_pixels * cm_per_pixel * 0.95  # Adjusted factor for accuracy
        measurements['Shoulder Circumference (cm)'] = round(shoulder_circumference_cm, 2)
    else:
        measurements['Shoulder Circumference (cm)'] = "N/A"


    # Thigh circumference
    left_knee = front_landmarks.get("Left Knee")
    if left_hip and left_knee:
        thigh_length_pixels = calculate_distance(left_hip, left_knee)
        thigh_circumference_cm = thigh_length_pixels * cm_per_pixel * 1.3  # Adjusted factor for accuracy
        measurements['Thigh Circumference (cm)'] = round(thigh_circumference_cm, 2)
    else:
        measurements['Thigh Circumference (cm)'] = "N/A"
    
    # Hips circumference
    if left_hip and right_hip:
        hips_width_pixels = calculate_distance(left_hip, right_hip)
        hips_circumference_cm = hips_width_pixels * cm_per_pixel * 2.7  # Adjusted factor for accuracy
        measurements['Hips Circumference (cm)'] = round(hips_circumference_cm, 2)
    else:
        measurements['Hips Circumference (cm)'] = "N/A"
    
    # Leg length measurement
    left_hip = front_landmarks.get("Left Hip")
    left_ankle = front_landmarks.get("Left Ankle")

    if left_hip and left_ankle:
        leg_length_pixels = calculate_distance(left_hip, left_ankle)
        leg_length_cm = leg_length_pixels * cm_per_pixel  # Adjust as needed based on your scale
        measurements['Leg Length (cm)'] = round(leg_length_cm, 2)
    else:
         measurements['Leg Length (cm)'] = "N/A"

    
    return measurements


## Function to validate measurements
def validate_measurements(measurements):
    valid_ranges = {
        'Neck Circumference (cm)': (30, 50),     
        'Chest Circumference (cm)': (80, 120),
        'Waist Circumference (cm)': (60, 110),
        'Wrist Circumference (cm)': (14, 25),
        'Arm Length (cm)': (50, 70),
        'Thigh Circumference (cm)': (40, 70),
        'Hips Circumference (cm)': (70, 120),
        'Leg Length (cm)': (70, 105),     
        'Shoulder Circumference (cm)': (30, 60),           
    }

    for measurement, value in measurements.items():
        if isinstance(value, (int, float)):
            low, high = valid_ranges.get(measurement, (None, None))
            if low is not None and (value < low or value > high):
                print(f"Warning: {measurement} of {value:.2f} cm is out of realistic range ({low}-{high} cm).")




# Function to save measurements to file
def save_measurements(measurements, user_info):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"measurements_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"=== Virtual Body Measurements ===\n")
        f.write(f"Age: {user_info.age}\n")
        f.write(f"Gender: {user_info.gender}\n")
        f.write(f"Height: {user_info.height_cm} cm\n")
        # f.write(f"Weight: {user_info.weight_kg} kg\n\n")
        
        f.write("=== Calculated Measurements ===\n")
        for measurement, value in measurements.items():
            f.write(f"{measurement}: {value}\n")
    
    print(f"\nMeasurements saved to {filename}")

# Function to handle live camera input and automated measurements
def main():
    user_info = get_user_info()
    detector = PoseDetector()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()
    
    # Set higher resolution for better visibility (portrait orientation)
    desired_width = 720  # Width less than height for portrait
    desired_height = 1280  # Height greater than width
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    
    # Create a resizable window
    window_name = "Virtual Body Measurement"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, desired_width, desired_height)
    
    # Define rectangle parameters for vertical window with larger size
    rect_width_ratio = 0.6  # 60% of the frame width
    rect_height_ratio = 0.8  # 80% of the frame height

    # Initialize capture states
    CAPTURE_FRONT = 'front'
    CAPTURE_SIDE = 'side'
    PROCESSING = 'processing'
    DONE = 'done'
    
    current_state = CAPTURE_FRONT
    capture_start_time = None
    capture_duration = 10  # seconds
    front_landmarks_accum = []
    side_landmarks_accum = []
    
    front_landmarks_final = None
    side_landmarks_final = None
    
    print("\n=== Instructions ===")
    print("The system will automatically capture your FRONT view for 10 seconds.")
    print("Please position yourself within the green box.")
    print("After FRONT view capture, rotate to your SIDE view.")
    print("The system will then automatically capture your SIDE view for 10 seconds.")
    print("Press 'q' at any time to quit.\n")
    
    # Initialize focal length to None
    focal_length_px = None
    
    while cap.isOpened() and current_state != DONE:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        frame = cv2.flip(frame, 1)  # Mirror image for better user experience
        frame_height, frame_width, _ = frame.shape
        
        # Calculate rectangle coordinates
        rect_start = (int((1 - rect_width_ratio ) / 2 * frame_width), int((1 - rect_height_ratio) / 2 * frame_height))
        rect_end = (int((1 + rect_width_ratio) / 2 * frame_width), int((1 + rect_height_ratio) / 2 * frame_height))
        rect_color = (0, 255, 0)  # Green
        rect_thickness = 4
        
        # Draw rectangle guide
        cv2.rectangle(frame, rect_start, rect_end, rect_color, rect_thickness)
        
        # Display state-specific instructions and countdown
        if current_state in [CAPTURE_FRONT, CAPTURE_SIDE]:
            if capture_start_time is None:
                capture_start_time = time.time()
                print(f"Starting {current_state.upper()} view capture.")
                if current_state == CAPTURE_FRONT:
                    # Calculate focal length in pixels based on initial frame
                    focal_length_px = calculate_focal_length_pixels(frame_width, user_info.camera_fov_deg)
                    print(f"Calculated Focal Length: {focal_length_px:.2f} pixels")
            
            elapsed_time = time.time() - capture_start_time
            remaining_time = int(capture_duration - elapsed_time)
            remaining_time = max(0, remaining_time)
            
            # Overlay text on the frame
            if current_state == CAPTURE_FRONT:
                cv2.putText(frame, f"Capturing FRONT view: {remaining_time} sec", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif current_state == CAPTURE_SIDE:
                cv2.putText(frame, f"Capturing SIDE view: {remaining_time} sec", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Detect landmarks
            landmarks = detector.detect_landmarks(frame)
            if landmarks:
                detector.draw_landmarks(frame, landmarks)
                landmarks_dict = extract_landmarks(landmarks, frame.shape)
                
                # Accumulate landmarks
                if current_state == CAPTURE_FRONT:
                    front_landmarks_accum.append(landmarks_dict)
                elif current_state == CAPTURE_SIDE:
                    side_landmarks_accum.append(landmarks_dict)
            
            # Display countdown
            cv2.putText(frame, f"Time Remaining: {remaining_time}s", 
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Check if capture duration is over
            if elapsed_time >= capture_duration:
                if current_state == CAPTURE_FRONT:
                    if front_landmarks_accum:
                        # Average landmarks
                        front_landmarks_final = average_landmarks(front_landmarks_accum)
                        print("Front view captured successfully.")
                    else:
                        print("No landmarks detected for Front view.")
                    # Transition to side view
                    current_state = CAPTURE_SIDE
                    capture_start_time = None
                elif current_state == CAPTURE_SIDE:
                    if side_landmarks_accum:
                        # Average landmarks
                        side_landmarks_final = average_landmarks(side_landmarks_accum)
                        print("Side view captured successfully.")
                    else:
                        print("No landmarks detected for Side view.")
                    # Transition to processing
                    current_state = PROCESSING
                    capture_start_time = None
        
        elif current_state == PROCESSING:
            # Process measurements
            if front_landmarks_final and side_landmarks_final and focal_length_px:
                measurements = calculate_measurements(front_landmarks_final, side_landmarks_final, user_info, focal_length_px)
                validate_measurements(measurements)
                
                # Display measurements
                print("\n=== Calculated Measurements ===")
                for measurement, value in measurements.items():
                    print(f"{measurement}: {value}")
                
                # Save measurements
                save_measurements(measurements, user_info)
            else:
                print("Error: Missing landmarks or focal length for measurement calculation.")
            
            current_state = DONE
        
        # Show the frame
        cv2.imshow(window_name, frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()


# Function to average landmarks over multiple frames
def average_landmarks(landmarks_list):
    averaged_landmarks = {}
    count = len(landmarks_list)
    if count == 0:
        return averaged_landmarks
    
    # Collect all landmark names
    all_landmarks = set()
    for landmarks in landmarks_list:
        all_landmarks.update(landmarks.keys())
    
    for landmark in all_landmarks:
        x_sum = 0
        y_sum = 0
        valid = 0
        for landmarks in landmarks_list:
            if landmark in landmarks:
                x_sum += landmarks[landmark][0]
                y_sum += landmarks[landmark][1]
                valid += 1
        if valid > 0:
            averaged_landmarks[landmark] = (int(x_sum / valid), int(y_sum / valid))
    
    return averaged_landmarks

if __name__ == "__main__":
    main()