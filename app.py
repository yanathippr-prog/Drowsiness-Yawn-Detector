# คำนวณทิศใบหน้า
#NOSE_TIP = 1
#EYE_LEFT_CENTER = 33
#EYE_RIGHT_CENTER = 263
#POSE_FRAMES = 15       
#pose_counter = 5       
#distract_total = 0         

            nose = face_landmarks[NOSE_TIP]
            le = face_landmarks[EYE_LEFT_CENTER]
            re = face_landmarks[EYE_RIGHT_CENTER]

            dist_to_left = np.abs(nose.x - le.x)
            dist_to_right = np.abs(nose.x - re.x)
            if dist_to_right == 0: dist_to_right = 0.001  
            turn_ratio = dist_to_left / dist_to_right

            eye_y_avg = (le.y + re.y) / 2.0
            vertical_drop = nose.y - eye_y_avg

            is_distracted = False
            status_text = "LOOKING FORWARD"
            status_color = (0, 255, 0) 

            if turn_ratio < 0.35:
                status_text = "LOOKING LEFT"
                is_distracted = True
                status_color = (0, 0, 255) 
            elif turn_ratio > 2.8:
                status_text = "LOOKING RIGHT"
                is_distracted = True
                status_color = (0, 0, 255)

            elif vertical_drop > 0.08: 
                status_text = "HEAD DOWN"
                is_distracted = True
                status_color = (0, 0, 255)

            if is_distracted:
                pose_counter += 1
                if pose_counter >= POSE_FRAMES:
                    cv2.putText(frame, f"!!! ALERT: {status_text} !!!", (30, 180), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    beep_distract() 
            else:
                if pose_counter >= POSE_FRAMES:
                    distract_total += 1
                pose_counter = 0

 # พิมพ์บอกสถานะการมองปัจจุบันที่มุมบนซ้าย (เปลี่ยนสีตามสถานะจริง)
            #cv2.putText(frame, f"STATUS: {status_text}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            # ป้ายไฟสรุปแต้มสะสมชิ้นใหม่ วางต่อท้ายจากแต้มเดิม (พิกัด Y = 480)
            #cv2.putText(frame, f"Total Distract: {distract_total}", (30, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)



# 1. ประกาศตัวควบคุมระบบเสียงตามสไตล์เดิม
if platform.system() == "Windows":
    import winsound
    def beep_drowsy():
        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
    def beep_distract():
        winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
else:
    import sys
    def beep_drowsy(): sys.stdout.write('\a'); sys.stdout.flush()
    def beep_distract(): sys.stdout.write('\a'); sys.stdout.flush()


# 2. ส่วนคำสั่งทดสอบรันเสียงจริง
print("กำลังทดสอบเสียงที่ 1: เสียงเตือนเมื่อหลับตา (SystemExclamation)")
beep_drowsy() # มีวงเล็บ () เพื่อสั่งให้ฟังก์ชันทำงานจริง

# สั่งให้ Python หยุดรอ 3 วินาที เพื่อให้เสียงเล่นจนจบเข้าหูฟัง ไม่ปิดโปรแกรมหนีไปก่อน
time.sleep(3) 

print("กำลังทดสอบเสียงที่ 2: เสียงเตือนเมื่อก้มหน้า/ไม่มองทาง (SystemAsterisk)")
beep_distract()

time.sleep(3)
print("สิ้นสุดการทดสอบระบบเสียง")