# https://blog.roboflow.com/yolov8-tracking-and-counting/
# https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-and-count-vehicles-with-yolov8.ipynb?ref=blog.roboflow.com#scrollTo=Q9ppb7bFvWfc
# https://www.youtube.com/watch?v=QV85eYOb7gk
# https://www.youtube.com/watch?v=IHbJcOex6dk

# Webcam

from ultralytics import YOLO # pip install ultralytics
#import numpy as np           # pip install numpy
#from PIL import Image        # pip install Pillow
#import cv2                   # pip install opencv-python
#from IPython import display  # pip install IPython
#display.clear_output()
import supervision # pip install supervision==0.1.0
print("supervision.__version__:", supervision.__version__)
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from datetime import datetime

# load model
model = YOLO('./models/klotsid_2_v8.pt')

#model.conf = 0.85
#model.iou = 0.2


# create instance of BoxAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=2, text_scale=1)

#frame = cv2.imread("images/WIN_20230427_12_57_45_Pro.jpg", cv2.IMREAD_COLOR)


# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [0, 1, 2, 3]

# predict
# model prediction on single frame and conversion to supervision Detections
#results = model.predict(source="0", show=True, stream=True)
results = model.predict(source="0", stream=True, show=True, conf=0.6)
print(*results)

"""
# Define your object classes
class_names = ['kolmne', 'Neljane', 'Viiene', 'Kuuene', 'Kahene', 'Ühene']

# Count the number of object classes detected
class_counts = {}
for res in results:
    if res is not None:
        print(res[0].boxes.cls.cpu().numpy().astype(int)) # Class id
        for obj in res:
            id = res[0].boxes.cls.cpu().numpy().astype(int)
            class_name = class_names[id[0]]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
print(class_counts)
#results.pandas().xyxy[0].value_counts('name')


detections = Detections(
    xyxy=results[0].boxes.xyxy.cpu().numpy(),
    confidence=results[0].boxes.conf.cpu().numpy(),
    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
)

# format custom labels
labels = [
    f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for _, confidence, class_id, tracker_id
    in detections
]

# annotate and display frame
frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)

now = datetime.now()
out_dir = r'./output/'
filename = out_dir + 'out_'+ str(now.hour) + str(now.minute) + '.jpg'

cv2.imshow("image", frame)
cv2.imwrite(filename, frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""