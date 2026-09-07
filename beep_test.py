import platform
import threading
from pygame import mixer
import time

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

# 2. ฟังก์ชันหลักที่เราเอาไปเรียกใช้ในเงื่อนไขลูป `while` 
def beep_drowsy():
    # แตกเธรดเพื่อไปสั่งเปิดไฟล์ mp3 แยกฉากหลัง ไม่ล็อกจอกล้องวิดีโอ
     threading.Thread(target=_play_mp3, args=("drowsy_alarm.wav",), daemon=True).start()
    
def beep_distract():
    threading.Thread(target=_play_mp3, args=("distract_alarm.wav",), daemon=True).start()

# 2. ส่วนคำสั่งทดสอบรันเสียงจริง
print("กำลังทดสอบเสียงที่ 1: เสียงเตือนเมื่อหลับตา (SystemExclamation)")
beep_drowsy() # มีวงเล็บ () เพื่อสั่งให้ฟังก์ชันทำงานจริง

# สั่งให้ Python หยุดรอ 3 วินาที เพื่อให้เสียงเล่นจนจบเข้าหูฟัง ไม่ปิดโปรแกรมหนีไปก่อน
time.sleep(3) 

print("กำลังทดสอบเสียงที่ 2: เสียงเตือนเมื่อก้มหน้า/ไม่มองทาง (SystemAsterisk)")
beep_distract()

time.sleep(3)
print("สิ้นสุดการทดสอบระบบเสียง")
