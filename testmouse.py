import sys
import math
import mediapipe as mp
import pyautogui
import cv2
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton
from pynput.mouse import Button, Controller

# настройки иекста для распознавания движения 
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,200)
fontScale = 2
fontColor = (255,255,255)
thickness = 4
lineType = 3

#Настройки управления
cofx = 1.6 #растяжение по Х
cofy = 1.9 #растяжение по Y
pos_y_move=300 #Сдвиг Y
pos_x_move=60 #Сдвиг Х
middle_distance=30 #расстояние ЛКМ Х2
ring_distance = 30 #расстояние ПКМ
finger_distance=30 #расстояние ЛКМ 


mouse=Controller()
class MainApp(QWidget):
    
    def __init__(self): #конструктор начального окна
        super().__init__()
        
        layout = QVBoxLayout()
        self.comboBox = QComboBox(self)
        for i in range(10):
            self.comboBox.addItem(str(i))
        self.function_selector = QComboBox(self)
        self.function_selector.addItem("Управление рукой")
        self.function_selector.addItem("Распознавание движения")
        self.start_button = QPushButton("Запустить")
        self.start_button.clicked.connect(self.start_selected_function)
        layout.addWidget(self.comboBox)
        layout.addWidget(self.function_selector)
        layout.addWidget(self.start_button)
        self.setLayout(layout)
        self.setWindowTitle("Stat panel")
        self.show()

    def start_selected_function(self): #выбор функции

        global chous
        chous = int(self.comboBox.currentText())
        selected_function = self.function_selector.currentText()
        if selected_function == "Управление рукой":
            self.start_hand_gesture_control()
        elif selected_function == "Распознавание движения":
            self.start_motion_detection_contour()
    
    def start_hand_gesture_control(self): #управление рукой
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        
        hands = mp_hands.Hands()
        cap = cv2.VideoCapture(chous)
        screen_width, screen_height = pyautogui.size()
        while cap.isOpened():
            ret, frame1 = cap.read()
            frame = cv2.flip(frame1, 1)
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    index_finger_landmark = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_finger_landmark = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_finger_landmark = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    thumb_landmark = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                    index_x, index_y = int(index_finger_landmark.x * screen_width), int(index_finger_landmark.y * screen_height)
                    middle_x, middle_y = int(middle_finger_landmark.x * screen_width), int(middle_finger_landmark.y * screen_height)
                    thumb_x, thumb_y = int(thumb_landmark.x * screen_width), int(thumb_landmark.y * screen_height)
                    ring_x, ring_y = int(ring_finger_landmark.x * screen_width), int(ring_finger_landmark.y * screen_height)

                    distance_index_thumb = math.hypot(index_x - thumb_x,index_y - thumb_y)
                    distance_middle_thumb = math.hypot(middle_x - thumb_x , middle_y - thumb_y)
                    distance_ring_thumb = math.hypot(ring_x - thumb_x , ring_y - thumb_y)
                    
                    print("указательный", distance_index_thumb)
                    print("средний", distance_middle_thumb)

                    if distance_index_thumb < finger_distance:
                        mouse.click(Button.left)
                        
                    elif distance_middle_thumb < middle_distance:
                        mouse.click(Button.left,2)
                    elif distance_ring_thumb < ring_distance:
                        mouse.click(Button.right)


                    

                    mouse.position=((thumb_x-pos_x_move)*cofx, (thumb_y-pos_y_move)*cofy)
                    print(f"x{mouse.position}")

                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Hand', frame)

            key = cv2.waitKey(1)
            if key == 27:
                break
            if cv2.getWindowProperty('Hand', cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()

    def start_motion_detection_contour(self): #распознавание движения
        cap = cv2.VideoCapture(chous)

        previous_frame = None
        movement_threshold = 500
        is_mirrored = True

        def alarm():
            cv2.putText(frame,'Motion!',bottomLeftCornerOfText,font,fontScale,fontColor,thickness,lineType)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if is_mirrored:
                frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if previous_frame is None:
                previous_frame = gray
                continue

            frame_delta = cv2.absdiff(previous_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = False

            for contour in contours:
                if cv2.contourArea(contour) < movement_threshold:
                    continue

                motion_detected = True

                
                cv2.drawContours(frame, [contour], -1, (0, 255, 4), 2)

            if motion_detected:
                alarm()
            cv2.imshow("Hand", frame)
            previous_frame = gray.copy()
            key = cv2.waitKey(1)
            if key == 27:
                break
            if cv2.getWindowProperty('Hand', cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApp()
    sys.exit(app.exec())