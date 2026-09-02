import cv2
import numpy as np # calculator
import mediapipe as mp
import platform
import threading
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

if platform.system() == "Windows":
    import winsound

    # สร้างตัวล็อกเช็กสถานะเสียง (ตั้งค่าเริ่มต้นเป็น False แปลว่ายังไม่มีเสียงดัง)
    sound_lock = False
    
    def _play_drowsy():
        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
        
    def _play_distract():
        winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
        
    # 2. ฟังก์ชันหลักที่เราจะเอาไปเรียกใช้ในลูป (แยกสมองวิ่งตัดหน้า ไม่รอเสียงจบ)
    def beep_drowsy():
        threading.Thread(target=_play_drowsy, daemon=True).start()
        
    def beep_distract():
        threading.Thread(target=_play_distract, daemon=True).start()
else:
    import sys
    def beep_drowsy(): sys.stdout.write('\a'); sys.stdout.flush()
    def beep_distract(): sys.stdout.write('\a'); sys.stdout.flush()

# --- ฟังก์ชันคำนวณระยะห่างระหว่างจุด 2 จุด (Euclidean Distance) ---
def get_dist(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# --- ฟังก์ชันคำนวณสัดส่วนดวงตา (Eye Aspect Ratio) ---
def calculate_ear(landmarks, eye_indices):
    # ระยะแนวดิ่ง 2 คู่
    p2_p6 = get_dist(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    p3_p5 = get_dist(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    # ระยะแนวนอน 1 คู่
    p1_p4 = get_dist(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    # สูตร EAR
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

# --- ฟังก์ชันคำนวณสัดส่วนริมฝีปาก (Mouth Aspect Ratio) ---
def calculate_mar(landmarks, mouth_indices):
    # ระยะแนวดิ่งของปากด้านใน 3 คู่ เพื่อความแม่นยำ
    d1 = get_dist(landmarks[mouth_indices[1]], landmarks[mouth_indices[6]])
    d2 = get_dist(landmarks[mouth_indices[2]], landmarks[mouth_indices[5]])
    d3 = get_dist(landmarks[mouth_indices[3]], landmarks[mouth_indices[4]])
    # ระยะแนวนอน (มุมปากซ้ายไปขวา)
    horizontal_dist = get_dist(landmarks[mouth_indices[0]], landmarks[mouth_indices[7]])
    return (d1 + d2 + d3) / (3.0 * horizontal_dist)

# --- ดัชนีจุด Landmark ของ MediaPipe Face Mesh ---
LEFT_EYE = [362, 385, 387, 263, 373, 380]   # จุดรอบตาซ้าย
RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # จุดรอบตาขวา
INNER_MOUTH = [78, 81, 82, 83, 313, 312, 311, 308] # จุดปากด้านใน
NOSE_TIP = 1
EYE_LEFT_CENTER = 33
EYE_RIGHT_CENTER = 263

# --- ตั้งค่าเกณฑ์กำหนด (Thresholds) ---
EAR_THRESHOLD = 0.21   # ต่ำกว่านี้แปลว่าหลับตา
MAR_THRESHOLD = 0.42   # สูงกว่านี้แปลว่ากำลังหาว
YAWN_FRAMES = 20       # ต้องอ้าปากกว้างติดต่อกันนาน 20 เฟรมขึ้นไป ถึงจะตัดสินว่า "หาว"
yawn_counter = 0       #ตัวนับเฟรมสะสมของการหาว
yawn_total = 0         # (ของแถม) ตัวแปรนับจำนวนครั้งที่หาวสะสมในโปรแกรม
CONSEC_FRAMES = 20      # ต้องหลับตาติดต่อกันกี่เฟรม ถึงจะเตือนว่า "ง่วงนอน"
blink_counter = 0
drowsy_total = 0       # แต้มสะสม: จำนวนครั้งที่หลับใน

#NOSE_TIP = 1
#EYE_LEFT_CENTER = 33
#EYE_RIGHT_CENTER = 263
#POSE_FRAMES = 15       
#pose_counter = 5       
#distract_total = 0  

# --- เตรียมระบบตรวจจับ ---
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break # [อันที่ 1] เช็กกล้องทันทีหลังจากอ่านภาพมา ถ้ากล้องพังให้หยุด

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    # แปลงสีภาพจาก BGR เป็น RGB เพื่อให้ MediaPipe ใช้งาน
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)
    
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            # 1. คำนวณค่า EAR ของตาทั้งสองข้างแล้วหาค่าเฉลี่ย
            left_ear = calculate_ear(face_landmarks, LEFT_EYE)
            right_ear = calculate_ear(face_landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            # 2. คำนวณค่า MAR ของปาก
            mar = calculate_mar(face_landmarks, INNER_MOUTH)

            # 3. ตรวจสอบการหลับตา (Drowsiness/Sleep Detection)
            if avg_ear < EAR_THRESHOLD:
                blink_counter += 1
                if blink_counter >= CONSEC_FRAMES:
                    cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if blink_counter == CONSEC_FRAMES:
                        beep_drowsy()
            else:
                if blink_counter >= CONSEC_FRAMES:
                    drowsy_total += 1
                blink_counter = 0  # รีเซ็ตตัวนับถ้าลืมตาขึ้นมาแล้ว
            # 4. ตรวจสอบการหาวแบบนับเฟรมค้าง (Yawning Detection with Frame Counter)
            if mar > MAR_THRESHOLD:
                yawn_counter += 1  # ถ้าปากกว้างเกินเกณฑ์ ให้บวกคะแนนเฟรมไปเรื่อย ๆ
                
                # ถ้าอ้าปากกว้างค้างไว้นานจนถึงจำนวนเฟรมที่ตั้งไว้
                if yawn_counter >= YAWN_FRAMES:
                    cv2.putText(frame, "!!! YAWNING DETECTED !!!", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
  
            else:
                # ของแถม: ถ้าหุบปากลงแล้ว และก่อนหน้านี้สถิติเฟรมถึงเกณฑ์แปลว่าหาวจบไป 1 ครั้ง
                if yawn_counter >= YAWN_FRAMES:
                    yawn_total += 1  # นับจำนวนครั้งที่หาวเพิ่มขึ้น 1 ครั้ง
                
                yawn_counter = 0  # รีเซ็ตตัวนับเฟรมทันทีเมื่อหุบปากปกติ เพื่อเริ่มนับใหม่รอบหน้า
 

            # แสดงค่าสถานะบนหน้าจอแบบเรียลไทม์
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # พ่นตัวเลขสถิติ yawn_total ออกทางหน้าจอ (แสดงมุมบนซ้าย ห่างขอบลงมาพิกเซลที่ 180)
            cv2.putText(frame, f"Total Drowsy: {drowsy_total}", (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Total Yawns: {yawn_total}", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # พิมพ์บอกสถานะการมองปัจจุบันที่มุมบนซ้าย (เปลี่ยนสีตามสถานะจริง)
            #cv2.putText(frame, f"STATUS: {status_text}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            # ป้ายไฟสรุปแต้มสะสมชิ้นใหม่ วางต่อท้ายจากแต้มเดิม (พิกัด Y = 480)
            #cv2.putText(frame, f"Total Distract: {distract_total}", (30, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)


    cv2.imshow('Drowsiness & Yawn Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()