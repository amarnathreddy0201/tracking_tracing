import logging.config
from ultralytics import YOLO
import queue
import cv2
import time
import pandas as pd
import threading
import concurrent.futures


import logging
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)



model_path = r"assets\yolov8_models\yolov8s.pt"
model = YOLO(model_path)

class VideoCapture:

    def __init__(self,camera_link):
        self.cap = cv2.VideoCapture(camera_link)
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)
        # self.cap.open(camera_link)

        self.q = queue.Queue()
        # threading.Timer(0, self._reader).start()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    
    def _reader(self):
        """
        Function to continuously read frames from the video capture object
        and put them in the queue for further processing.
        """
        try :
            while True:
                # Read a frame from the video capture object
                ret, frame = self.cap.read()

                # If no frame is returned, break the loop
                if not ret:
                    break

                # If the queue is not empty, try to get an item from it without
                # blocking. If the queue is empty, pass
                if not self.q.empty():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                
                if self.q.qsize() < 10:
                    # Put the frame in the queue
                    self.q.put(frame)

        except Exception as e:
            logger.error(f"Video Capture error : {e}")

    def read(self):
        return self.q.get()
    


def camera_streamin():

    start_time = time.time()
    camera_link =0 
    cap = VideoCapture(camera_link=camera_link)
    display_time = 1
    fc = 0

    start = 0
    person_count = 0  # Initialize person count
    while True:
        TIME = time.time() - start_time
        if TIME >= display_time:
            FPS = fc / TIME
            fc = 0
            start_time = time.time()
            FPS = round(FPS)
            fps_disp = "FPS: " + str(FPS)[:5]

        frame = cap.read()
        frame = cv2.resize(frame,(640,480))

        results = model(frame)

        if results and len(results[0].boxes) > 0:
            person = False
            vehicle = False
            
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

                        person_count += 1

        # Display person count on the frame
        cv2.putText(frame, f"Persons: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break


def Streams_Parallel():
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        
    
        results = [executor.submit(camera_streamin)]
        
    for _ in concurrent.futures.as_completed(results):
        pass

if __name__ == "__main__":
    Streams_Parallel()

