import platform
import time

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
