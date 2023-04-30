# Yolo basic webcam
# Tauno Erik
# 28.04.2023
# Source: https://www.youtube.com/watch?v=WgPbbWmnXJ8

from ultralytics import YOLO
import cv2
import cvzone
import math

MODEL_FILE = '../models/klotsid_2_v4.pt'

EST = 0
ENG = 1

LANG = ENG # Sellect language

class_names = [
    ['Kolmene', 'Neljane', 'Viiene', 'Kuuene'],
    ['Three', 'Four', 'Five', 'Six']
    ]

model = YOLO(MODEL_FILE)
#model.fuse()

cap = cv2.VideoCapture(0) # webcam
cap.set(3, 1280) # with
cap.set(4, 720)  # height

# Video file
#cap = cv2.VideoCapture('../images/WIN_20230427_13_29_35_Pro.mp4')

while True:
    success, frame = cap.read()
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # v1 - cv2
            #x1, y1, x2, y2 = box.xyxy[0]
            #x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 3)

            # v2 - cvzone
            #x1, y1, w, h = box.xywh[0]
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(frame, (x1,y1,w,h))

            #conf = math.ceil((box.conf[0]*100))/100 # round
            #cvzone.putTextRect(frame, f'{conf}', (max(0,x1), max(30, y1)) )

            name_id = int(box.cls[0])
            cvzone.putTextRect(frame,
                               f'{class_names[LANG][name_id]}',
                               (max(0,x1),
                               max(30, y1)),
                               scale=1.5,
                               thickness=1)



    cv2.imshow("Camera", frame)
    cv2.waitKey(1)
