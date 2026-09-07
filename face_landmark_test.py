import cv2
import numpy as np # calculator
import mediapipe as mp
import platform
import threading
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pygame import mixer
# สั่งให้ระบบมิกเซอร์เสียงเริ่มต้นทำงาน
mixer.init()

# 1. ฟังก์ชันดิบสำหรับเรียกเปิดไฟล์มัลติมีเดียเบื้องหลัง
def _play_mp3(file_path):
    try:
        # ใช้ระบบแชนเนลแยกเพื่อเล่นเสียงสั้นฉับไว (Sound Object) ไม่กวนเพลงหลัก
        sound = mixer.Sound(file_path)
        # 📌 เพิ่มบรรทัดนี้: สั่งตั้งค่าความดัง (ใส่ค่าระหว่าง 0.0 ถึง 1.0)
        sound.set_volume(1.0) 
        sound.play()
    except Exception as e:
        print(f"ระบบเสียงติดขัด: {e}")

def beep_drowsy():
    # แตกเธรดเพื่อไปสั่งเปิดไฟล์ mp3 แยกฉากหลัง ไม่ล็อกจอกล้องวิดีโอ
     threading.Thread(target=_play_mp3, args=("drowsy_alarm.wav",), daemon=True).start()

def beep_distract():
    threading.Thread(target=_play_mp3, args=("distract_alarm.wav",), daemon=True).start()

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
CALIBRATION_FRAMES = 150  # เก็บข้อมูล 150 เฟรมแรก (ประมาณ 5 วินาทีที่ 30 FPS)
frame_count = 0           # ตัวนับเฟรมปัจจุบันเพื่อเช็กว่าพ้นช่วงสอบเทียบหรือยัง
calib_ear_list = []       # ลิสต์เก็บค่า EAR ตอนลืมตาปกติ
calib_mar_list = []       # ลิสต์เก็บค่า MAR ตอนหุบปากปกติ

EAR_THRESHOLD = 0.21   # ต่ำกว่านี้แปลว่าหลับตา
MAR_THRESHOLD = 0.42   # สูงกว่านี้แปลว่ากำลังหาว

CONSEC_FRAMES = 20      # ต้องหลับตาติดต่อกันกี่เฟรม ถึงจะเตือนว่า "ง่วงนอน"
BLINKLESS_THRESHOLD = 240  # ต้องลืมตาค้างนานติดต่อกันเกิน 240 เฟรม (ประมาณ 8 วินาทีที่ 30 FPS)
YAWN_FRAMES = 20       # ต้องอ้าปากกว้างติดต่อกันนาน 20 เฟรมขึ้นไป ถึงจะตัดสินว่า "หาว"

blink_counter = 0
staring_counter = 0        # ตัวนับเฟรมสะสมของการลืมตาค้าง
staring_total = 0          # แต้มสะสม: จำนวนครั้งที่เหม่อลอยค้าง
yawn_counter = 0
yawn_total = 0         # (ของแถม) ตัวแปรนับจำนวนครั้งที่หาวสะสมในโปรแกรม
drowsy_total = 0       # แต้มสะสม: จำนวนครั้งที่หลับใน

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
                if staring_counter >= BLINKLESS_THRESHOLD:
                    staring_total += 1
                staring_counter = 0
                if blink_counter >= CONSEC_FRAMES:
                    cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if blink_counter == CONSEC_FRAMES :
                        beep_drowsy()
            else:
                if blink_counter >= CONSEC_FRAMES:
                    drowsy_total += 1
                blink_counter = 0  # รีเซ็ตตัวนับถ้าลืมตาขึ้นมาแล้ว
                staring_counter += 1
                if staring_counter >= BLINKLESS_THRESHOLD:
                    cv2.putText(frame, "!!! DISTRACTION: EYE STARING !!!", (30, 180), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    # ส่งเสียงเตือนคีย์ทุ้ม (หรือคีย์แยก) เฉพาะเฟรมแรกที่เข้าเกณฑ์เหม่อลอย
                    if staring_counter == BLINKLESS_THRESHOLD:
                        beep_distract()
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
            cv2.putText(frame, f"Total Distract: {staring_total}", (30, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)

    cv2.imshow('Drowsiness & Yawn Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()