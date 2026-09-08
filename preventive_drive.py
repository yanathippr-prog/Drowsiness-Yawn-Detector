import cv2
import numpy as np # calculator
import mediapipe as mp
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pygame import mixer
import time

mixer.init()

# 1. ฟังก์ชันดิบสำหรับเรียกเปิดไฟล์มัลติมีเดียเบื้องหลัง
def _play_mp3(file_path):
    try:
        # ใช้ระบบแชนเนลแยกเพื่อเล่นเสียงสั้นฉับไว (Sound Object) ไม่กวนเพลงหลัก
        sound = mixer.Sound(file_path)
        # สั่งตั้งค่าความดัง (ใส่ค่าระหว่าง 0.0 ถึง 1.0)
        sound.set_volume(1.0) 
        channel = sound.play()
        if file_path == "drowsy_alarm.wav":
            time.sleep(0.3)  # ปรับความยาวเสียงหลับตาตรงนี้ (0.3 วินาที)
            channel.stop()   # ตัดจบเสียงทันที

    except Exception as e:
        print(f"ระบบเสียงติดขัด: {e}")

def beep_drowsy():
    # แตกเธรดเพื่อไปสั่งเปิดไฟล์ wav แยกฉากหลัง ไม่ล็อกจอกล้องวิดีโอ
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
CALIBRATION_FRAMES = 150  # เก็บข้อมูล 150 เฟรมแรก (ประมาณ 5 วินาที)
frame_count = 0           # ตัวนับเฟรมปัจจุบันเพื่อเช็กว่าพ้นช่วง calibration รึยัง
calib_ear_list = []       # ลิสต์เก็บค่า EAR ตอนลืมตาปกติ
calib_mar_list = []       # ลิสต์เก็บค่า MAR ตอนหุบปากปกติ

EAR_THRESHOLD = 0.21   # ต่ำกว่านี้แปลว่าหลับตา
MAR_THRESHOLD = 0.42   # สูงกว่านี้แปลว่ากำลังหาว

CONSEC_FRAMES = 20      # ต้องหลับตาติดต่อกันกี่เฟรม ถึงจะเตือนว่า "ง่วงนอน"
BLINKLESS_THRESHOLD = 240  # ต้องลืมตาค้างนานติดต่อกันเกิน 240 เฟรม (ประมาณ 8 วินาทีที่ 30 FPS)
YAWN_FRAMES = 15       # ต้องอ้าปากกว้างติดต่อกันนาน 15 เฟรมขึ้นไป ถึงจะตัดสินว่า "หาว"

blink_counter = 0
drowsy_total = 0       # แต้มสะสม: จำนวนครั้งที่หลับใน
drowsy_total = 0       # แต้มสะสม: จำนวนครั้งที่หลับตา
yawn_counter = 0       # ตัวนับเฟรมสะสมของการอ้าปากค้าง
yawn_total = 0         # แต้มสะสม: จำนวนครั้งที่หาว
staring_counter = 0        # ตัวนับเฟรมสะสมของการลืมตาค้าง
staring_total = 0          # แต้มสะสม: จำนวนครั้งที่เหม่อลอยค้าง

# --- ตั้งค่าเกณฑ์ตรวจจับการไม่มองทาง (Distraction / Looking Away) ---
# จุดขอบหน้าซ้ายและขวาของ MediaPipe Face Mesh
FACIAL_LEFT_EDGE = 234
FACIAL_RIGHT_EDGE = 454

POSE_FRAMES = 75       # ต้องหันหน้าหนีค้างเกิน 75 เฟรม (ประมาณ 2.5 วินาที) ถึงจะเตือน
pose_counter = 0       # ตัวนับเฟรมสะสมของการไม่มองถนน
distract_total = 0     # แต้มสะสม: จำนวนครั้งที่ไม่มองทางรวม


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

            

            frame_count += 1
            if frame_count <= CALIBRATION_FRAMES:
                # ช่วง 5 วินาทีแรก: บันทึกข้อมูลพิกัดชีวภาพเข้าคลังดาต้าเบส
                calib_ear_list.append(avg_ear)
                calib_mar_list.append(mar)
                
                # แสดงผลคำแนะนำบนหน้าจอให้คนขับมองตรงนิ่ง ๆ
                cv2.putText(frame, f"CALIBRATING BASELINE: {int((frame_count/CALIBRATION_FRAMES)*100)}%", (30, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "Please look forward with normal eyes...", (30, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                continue
            elif frame_count == CALIBRATION_FRAMES + 1:
                # 🧠 จังหวะผ่านวินาทีที่ 5: สั่งให้คณิตศาสตร์สถิติวินิจฉัยคำนวณหาเกณฑ์เฉพาะบุคคล (Threshold Matrix)
                mean_ear = np.mean(calib_ear_list)  # หาค่าเฉลี่ยสัดส่วนตาสภาวะปกติ
                mean_mar = np.mean(calib_mar_list)  # หาค่าเฉลี่ยสัดส่วนปากสภาวะปกติ
                
                # นำมาหักลบส่วนต่างทางชีววิทยาเพื่อสร้างโมเดลเกณฑ์ตัดแต้ม (Adaptive Calibration Model)
                EAR_THRESHOLD = mean_ear * 0.75     # คำนวณว่าถ้าตาหรี่ลงเหลือ 75% ของตาปกติ แปลว่าหลับตา
                MAR_THRESHOLD = mean_mar + 0.05     # คำนวณว่าถ้าปากอ้ากว้างเกิน 2.2 เท่าของปากปกติ แปลว่าหาว
                print(f"[ML Configured] EAR THRESHOLD: {EAR_THRESHOLD:.2f} | MAR THRESHOLD: {MAR_THRESHOLD:.2f} | EAR: {mean_ear:.2f} | MAR: {mean_mar:.2f}")

            # 3. ตรวจสอบการหลับตา (Drowsiness/Sleep Detection)
            if avg_ear < EAR_THRESHOLD:
                blink_counter += 1
                if staring_counter >= BLINKLESS_THRESHOLD:
                    staring_total += 1
                staring_counter = 0
                if blink_counter >= CONSEC_FRAMES:
                    cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (30, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if (blink_counter - CONSEC_FRAMES) % 10 == 0 :
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
                    cv2.putText(frame, "!!! YAWNING DETECTED !!!", (30, 140), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                if yawn_counter == YAWN_FRAMES: beep_drowsy()
            #elif mean_mar <= mar < MAR_THRESHOLD and avg_ear < (mean_ear * 0.92):
                #yawn_counter += 1
                #if yawn_counter >= YAWN_FRAMES:
                    #cv2.putText(frame, "!!! YAWNING (HAND COVERED) !!!", (30, 130), 
                                #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                    #if yawn_counter == YAWN_FRAMES: beep_drowsy()
            else:
                # ของแถม: ถ้าหุบปากลงแล้ว และก่อนหน้านี้สถิติเฟรมถึงเกณฑ์แปลว่าหาวจบไป 1 ครั้ง
                if yawn_counter >= YAWN_FRAMES:
                    yawn_total += 1  # นับจำนวนครั้งที่หาวเพิ่มขึ้น 1 ครั้ง
                
                yawn_counter = 0  # รีเซ็ตตัวนับเฟรมทันทีเมื่อหุบปากปกติ เพื่อเริ่มนับใหม่รอบหน้า
                        # 3. ตรรกะตรวจจับการไม่มองทาง (Distraction Detection)
            nose = face_landmarks[NOSE_TIP]
            f_left = face_landmarks[FACIAL_LEFT_EDGE]
            f_right = face_landmarks[FACIAL_RIGHT_EDGE]

            # คำนวณระยะห่างระหว่างจมูกไปยังขอบหน้าซ้ายและขวา
            dist_to_left_edge = np.abs(nose.x - f_left.x)
            dist_to_right_edge = np.abs(nose.x - f_right.x)
            if dist_to_right_edge == 0: dist_to_right_edge = 0.001
            
            # อัตราส่วนการหันหน้าสากล
            face_turn_ratio = dist_to_left_edge / dist_to_right_edge

            # กำหนดเกณฑ์ตัดสิน (ถ้าหันซ้ายค่าจะต่ำกว่า 0.45 ถ้าหันขวาค่าจะพุ่งเกิน 2.2)
            if face_turn_ratio < 0.45 or face_turn_ratio > 2.2:
                pose_counter += 1
                if pose_counter >= POSE_FRAMES:
                    cv2.putText(frame, "!!! ALERT: LOOKING AWAY !!!", (30, 180), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    if pose_counter == POSE_FRAMES:
                        beep_distract()
            else:
                # พอบิดหน้ากลับมามองตรง ให้บวกแต้มสะสม 1 ครั้ง และรีเซ็ตตัวนับเป็น 0
                if pose_counter >= POSE_FRAMES:
                    distract_total += 1
                pose_counter = 0

 

            # แสดงค่าสถานะบนหน้าจอแบบเรียลไทม์
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # พ่นตัวเลขสถิติ yawn_total ออกทางหน้าจอ (แสดงมุมบนซ้าย ห่างขอบลงมาพิกเซลที่ 180)
            cv2.putText(frame, f"Total Drowsy: {drowsy_total}", (30, 440), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Total Yawns: {yawn_total}", (30, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Total Distract: {staring_total}", (30, 480), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Face Turn Ratio: {face_turn_ratio:.2f}", (30, 280), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    cv2.imshow('Drowsiness & Yawn Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()