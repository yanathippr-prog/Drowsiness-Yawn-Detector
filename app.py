# คำนวณทิศใบหน้า
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