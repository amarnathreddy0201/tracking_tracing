from collections import OrderedDict
import time

import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO
import concurrent.futures
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import time



from main import VideoCapture
model = YOLO(r"assets\yolov8_models\yolov8s.pt")

from collections import OrderedDict
import numpy as np
import time


import logging
logging.basicConfig(
    filename = "logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.entry_time = {}
        self.exit_time = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid, current_time):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.entry_time[self.next_object_id] = current_time
        self.next_object_id += 1

    def deregister(self, object_id, current_time):
        del self.objects[object_id]
        del self.disappeared[object_id]
        entry_time = self.entry_time.pop(object_id)
        self.exit_time[object_id] = current_time
        return current_time - entry_time

    def update(self, rects, current_time):
        """
        weghpiuwehgu
        """
        time_present = {}
        try:
            if len(rects) == 0:
                for object_id in list(self.disappeared.keys()):
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        time_present[object_id] = self.deregister(object_id, current_time)
                return self.objects, time_present

            input_centroids = np.zeros((len(rects), 2), dtype="int")
            for (i, (startX, startY, endX, endY)) in enumerate(rects):
                cX = int((startX + endX) / 2.0)
                cY = int((startY + endY) / 2.0)
                input_centroids[i] = (cX, cY)

            if len(self.objects) == 0:
                for i in range(0, len(input_centroids)):
                    self.register(input_centroids[i], current_time)
            else:
                object_ids = list(self.objects.keys())
                object_centroids = list(self.objects.values())

                D = dist.cdist(np.array(object_centroids), input_centroids)

                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                used_rows = set()
                used_cols = set()

                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue

                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.disappeared[object_id] = 0

                    used_rows.add(row)
                    used_cols.add(col)

                unused_rows = set(range(0, D.shape[0])).difference(used_rows)
                unused_cols = set(range(0, D.shape[1])).difference(used_cols)

                if D.shape[0] >= D.shape[1]:
                    for row in unused_rows:
                        object_id = object_ids[row]
                        self.disappeared[object_id] += 1
                        if self.disappeared[object_id] > self.max_disappeared:
                            time_present[object_id] = self.deregister(object_id, current_time)
                else:
                    for col in unused_cols:
                        self.register(input_centroids[col], current_time)

            return self.objects, time_present

        except Exception as error:
            logger.error(error)
            return self.objects, time_present

def camera_streamin():
    # Initialize the centroid tracker
    ct = CentroidTracker(max_disappeared=10)

    start_time = time.time()
    camera_link = 0 
    cap = VideoCapture(camera_link=camera_link)
    display_time = 1
    fc = 0

    start = 0
    time_present = {}
    while True:
        TIME = time.time() - start_time
        if TIME >= display_time:
            FPS = fc / TIME
            fc = 0
            start_time = time.time()
            FPS = round(FPS)
            fps_disp = "FPS: " + str(FPS)[:5]

        frame = cap.read()
        frame = cv2.resize(frame, (640, 480))

        results = model(frame)

        if results and len(results[0].boxes) > 0:
            rects = []

            a = results[0].boxes.cpu().numpy().data
            px = pd.DataFrame(a).astype("float")
            number_of_objects = len(px)

            for result in results:
                for i in range(number_of_objects):
                    # try:
                    box = result.boxes.xywh[i].cpu().numpy()

                    # use it to find the center
                    # x, y, width, height = map(int, box)

                    cls = int(result.boxes.cls[i].item())
                    name = result.names[cls]


                    if name == "person":
                        person_x, person_y, person_width, person_height = map(int, box)
                        # cv2.circle(frame, (person_x, person_y),5,(255, 255, 255),-1)
                        # =======================================
                        person_dimentions = (person_width,person_height)
                        center_x , centery = person_x - person_width/2 , person_y - person_height/2
                        cv2.rectangle(frame, (person_x - person_width // 2, person_y - person_height // 2),
                                        (person_x + person_width // 2, person_y + person_height // 2), (0, 255, 255),2)
                        cv2.line(frame, (0, 0), (int(person_x), int(person_y)), (0, 255, 0), 2)

                        endX,endY= person_x + person_width  , person_y + person_height
                        rects.append((person_x, person_y, endX, endY))
                        cv2.line(frame, (0, 0), (int(person_x), int(person_y)), (0, 255, 0), 2)

            objects, time_present_tmp = ct.update(rects, time.time())

            # If time_present_tmp is not None, update time_present
            if time_present_tmp is not None:
                for object_id, time_spent in time_present_tmp.items():
                    if object_id not in time_present:
                        time_present[object_id] = 0
                    time_present[object_id] += time_spent

            # Count unique persons
            person_count = len(objects)

            for object_id, centroid in objects.items():
                # Draw centroid
                cv2.circle(frame, centroid, 4, (0, 255, 0), -1)

        # Display person count and time each person is present on the frame
        cv2.putText(frame, f"Persons: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for object_id, time_spent in time_present.items():
            cv2.putText(frame, f"Person {object_id}: {time_spent} frames", (10, 70 + 30 * object_id), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

def Streams_Parallel():
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            results = [executor.submit(camera_streamin)]
            
        for _ in concurrent.futures.as_completed(results):
            pass
    except Exception as error:
        logger.error(error)

if __name__ == "__main__":
    Streams_Parallel()