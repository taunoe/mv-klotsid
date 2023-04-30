# Yolo webcam, img mask, traking
# # Tracks only one class !!
# Tauno Erik
# 29.04.2023
# Source: https://www.youtube.com/watch?v=WgPbbWmnXJ8
# Install nvidia-cuda-toolkit
# visual studio communitu (on win)

from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *  # Download sort.py https://github.com/abewley/sort

MODEL_FILE = '../models/klotsid_2_v5.pt'
CONF_TH = 0.8

EST = 0
ENG = 1
LANG = EST  # Select language

class_names = [
    ['Kolmene', 'Neljane', 'Viiene', 'Kuuene'],
    ['Three', 'Four', 'Five', 'Six']
]

# https://github.com/ultralytics/yolov5/blob/4bb7eb8b849fc8a90823a60e2b7a8ec9e38926bf/utils/plots.py#L31-L47
colors = [(199,55,255), (56,56,255), (134,219,61), (168,153,44),(151,157,255), (52,147,26), (31,112,255), (29,178,255), (49,210,207), (10,249,72), (23,204,146)]

lang_txt = [
    ['Kokku: ', 'Kaamera', 'Loendur'],
    ['Total: ', 'Camera', 'Counter']
]

mask = cv2.imread('mask.png')  # sektor, kus loeme

LOGO_FILE = 'logo.png'

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)



model = YOLO(MODEL_FILE)
model.fuse()

cap = cv2.VideoCapture(0)  # webcam
WIDTH = 1280
HEIGHT = 720
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

line_pt = [int((WIDTH/2)), 50, int((WIDTH/2)), (HEIGHT-50)] # Loendus joon x1

# Video file
#cap = cv2.VideoCapture('../images/WIN_20230427_13_29_35_Pro.mp4')

# Count the number of object classes detected
total_counter = []  # Kokku kõiki erinevaid
class_counts = {}   # Class: count

while True:
    success, frame = cap.read()
    img_region = cv2.bitwise_and(frame, mask)

    # Add logo
    #logo = cv2.imread(LOGO_FILE, cv2.IMREAD_UNCHANGED)
    #frame = cvzone.overlayPNG(frame, logo, (0,0))

    results = model(img_region, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            conf = math.ceil((box.conf[0]*100))/100 # round
            #cvzone.putTextRect(frame, f'{conf}', (max(0,x1), max(30, y1)) )

            name_id = int(box.cls[0]) # class
            current_name = class_names[LANG][name_id]
            
            if conf > CONF_TH:
                # Track
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array)) # add to tracker

                #Draw rectangel
                #cvzone.cornerRect(frame, (x1,y1,w,h), l=25, rt=5)
                cv2.rectangle(frame, (x1,y1), (x2,y2),colors[name_id], 3)
                # Write Name:
                cvzone.putTextRect(frame,f'{current_name}',(max(0,x1), max(30, y1)),scale=1.5,thickness=1,offset=5,colorR=colors[name_id])

    results_tracker = tracker.update(detections)

    # Draw counting line
    cv2.line(frame, (line_pt[0], line_pt[1]), (line_pt[2], line_pt[3]), (0,200,0), 3)
    # Write Counter Title
    cvzone.putTextRect(frame,
                       f'{lang_txt[LANG][2]}',
                       (int((WIDTH/2))-68, 40),
                       scale=2,
                       thickness=2,
                       offset=10,
                       colorR=(0,200,0)
                       )

    for result in results_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        #print(result)
        w, h = x2-x1, y2-y1
        #cvzone.cornerRect(frame, (x1,y1,w,h), l=25, rt=2, colorR=(255,0,0))
        #cv2.rectangle(frame, (x1,y1), (x2,y2),(0,0,200), 3)

        # Write id
        # cvzone.putTextRect(frame,f'id{id}',(max(0,x1),max(30, y1)),scale=1.5,thickness=1,offset=5)
        
        # object center
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(frame, (cx, cy), 5, (255,255,0), cv2.FILLED)

        # Count all objects
        if line_pt[1] < cy < line_pt[3] and line_pt[0]-20 < cx < line_pt[2]+20:
            if total_counter.count(id) == 0: # kui ei ole juba loetud
                total_counter.append(id)
                # Blink line color to red
                cv2.line(frame, (line_pt[0], line_pt[1]), (line_pt[2], line_pt[3]), (0,0,200), 3)
                
                # Count classes separately
                # Tracker toetab ühte Klassi
                # result ainult sellest piirkonnast?
                count_mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.rectangle(count_mask, ((x1-15), (y1-15)), ((x2+15), (y2+15)), 255, -1)
                #cv2.imshow("Rectangular Mask", count_mask)
                count_region = cv2.bitwise_and(img_region, img_region,mask=count_mask)
                #cv2.imshow("Masked", count_region)
                count_results = model(count_region)
                for cr in count_results:
                    cboxes = cr.boxes
                    for cbox in cboxes:
                        _id = int(cbox.cls[0]) # class
                        #_conf = math.ceil((box.conf[0]*100))/100
                        if _id in class_counts:
                            class_counts[_id] += 1
                        else:
                            class_counts[_id] = 1
               
            else:
                pass # Selle id-ga klots on juba loetud

    
    # Total counter text
    cvzone.putTextRect(frame,
                       f'{lang_txt[LANG][0]}{len(total_counter)}',
                       (WIDTH-200, HEIGHT-50),
                       scale=2,
                       thickness=2,
                       offset=10,
                       colorR=(20,20,20)
                    )
    # Name counts
    y_start = 20
    y_inc = 50
    y = y_start

    # Write each line of the text on the image
    for i, (key, value) in enumerate(class_counts.items()):
        old_y = y
        y = old_y + y_inc
        text = f'{class_names[LANG][key]}: {value}'
        #org = (10, y)
        cvzone.putTextRect(frame,
                           text,
                           (WIDTH-200, HEIGHT-50-y),
                           scale=2,
                           thickness=2,
                           offset=10,
                           colorR=colors[key]
                           )


    cv2.imshow(str(lang_txt[LANG][1]), frame)
    cv2.waitKey(1)
