import cv2
import numpy as np
import pyautogui

def detect_eyes(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        return eyes
    return []

def get_pupil_center(eye_frame):
    _, threshold = cv2.threshold(eye_frame, 45, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    cap = cv2.VideoCapture(0)
    screen_width, screen_height = pyautogui.size()
    
    gaze_duration = 0
    last_gaze_point = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        eyes = detect_eyes(frame, face_cascade, eye_cascade)
        
        if len(eyes) == 2:
            for (ex,ey,ew,eh) in eyes:
                eye_frame = frame[ey:ey+eh, ex:ex+ew]
                eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
                pupil_center = get_pupil_center(eye_frame)
                
                if pupil_center:
                    gaze_x = int((pupil_center[0] / ew) * screen_width)
                    gaze_y = int((pupil_center[1] / eh) * screen_height)
                    
                    if last_gaze_point == (gaze_x, gaze_y):
                        gaze_duration += 1
                    else:
                        gaze_duration = 0
                    
                    last_gaze_point = (gaze_x, gaze_y)
                    
                    screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
                    if gaze_duration > 30:
                        cv2.circle(screen, (gaze_x, gaze_y), 20, (0, 255, 0), -1)
                    else:
                        cv2.circle(screen, (gaze_x, gaze_y), 20, (0, 255, 0), 2)
                    
                    cv2.imshow('Screen', screen)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
