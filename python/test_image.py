# https://blog.roboflow.com/yolov8-tracking-and-counting/
# https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-and-count-vehicles-with-yolov8.ipynb?ref=blog.roboflow.com#scrollTo=Q9ppb7bFvWfc
# https://www.youtube.com/watch?v=QV85eYOb7gk

# Sisend: Pilt
# Tauno Erik
# Muudetud: 27.04.2023
# Use 'q' to close window!

from ultralytics import YOLO # pip install ultralytics
import cv2                   # pip install opencv-python

#from IPython import display  # pip install IPython
#display.clear_output()
import supervision           # pip install supervision==0.1.0
#print("supervision.__version__:", supervision.__version__)
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

from datetime import datetime
import os, sys

# load model
model = YOLO('./models/klotsid_2_v5.pt')
#model.conf = 0.85
#model.iou = 0.2
model.fuse()

# create instance of BoxAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=2, text_scale=1)

frame = cv2.imread("./images/WIN_20230427_10_31_00_Pro.jpg", cv2.IMREAD_COLOR)

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [0, 1, 2, 3]

# predict
# model prediction on single frame and conversion to supervision Detections
results = model.predict(frame, conf=0.6)

# Define your object classes
class_names = ['Kolmene', 'Neljane', 'Viiene', 'Kuuene', 'Kahene', 'Ühene']

class_names_eng = {0:'Three', 1:'Four', 2:'Five', 3:'Six', 4:'Two', 5:'One'}

# Count the number of object classes detected
class_counts = {}

detections = Detections(
    xyxy=results[0].boxes.xyxy.cpu().numpy(),
    confidence=results[0].boxes.conf.cpu().numpy(),
    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
)

for class_id in detections.class_id:
    if class_names_eng[class_id] in class_counts:
        val = class_counts[class_names_eng[class_id]] + 1
        class_counts[class_names_eng[class_id]] = val
    else:
        class_counts[class_names_eng[class_id]] = 1


# format custom labels
labels = [
    #f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    #f"{CLASS_NAMES_DICT[class_id]}" #nimed mudelist
    f"{class_names_eng[class_id]}"
    for _, confidence, class_id, tracker_id
    in detections
]

# annotate and display frame
frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)

# font
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (10, 255, 255)
line_type = cv2.LINE_AA
# Get the height of the text
text_size = cv2.getTextSize('Sample', font, font_scale, 1)
(height, _) = text_size[0]

y_start = 20
y_inc = 50
y = y_start

# Write each line of the text on the image
for i, (key, value) in enumerate(class_counts.items()):
    old_y = y
    y = old_y + y_inc
    text = f'{key}: {value}'
    text_height = (i + 1) * height
    org = (10, y)
    cv2.putText(frame, text, org,
                font, font_scale, font_color, 2, line_type)

# Save image
now = datetime.now()
out_dir = r'./output/'
filename = out_dir + 'out_'+ str(now.hour) + str(now.minute) + '.jpg'

cv2.imshow("image", frame)
cv2.imwrite(filename, frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
sys.exit()
