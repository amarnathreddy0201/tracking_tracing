import cv2
import time
from ultralytics import YOLO
# from cv2 import VideoCapture
import cv2
import queue
import threading
import concurrent.futures


import logging
# model = YOLO("yolov8n.pt")
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

CLASS_NAMES = {
    0: "drone",
}


model = YOLO(r"drone.pt")


class Videocapture:

    def __init__(self,camera_link):
        self.camera_link = camera_link
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

                if not self.cap.isOpened():
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.camera_link)
                    continue

                    
                # Read a frame from the video capture object
                ret, frame = self.cap.read()
                
                if not ret:
                    self.cap = cv2.VideoCapture(self.camera_link)
                    
                    
                    continue

                if self.q.full():
                    try:
                        self.q.get_nowait()  # Remove the oldest frame to make space
                    except queue.Empty:
                        pass

                self.q.put(frame)
                time.sleep(0.01)
        

        except Exception as e:
            self.cap.release()
            cv2.destroyAllWindows()
            logger.error(f"Video Capture error : {e}")

    def read(self):
        # return self.q.get()

        try:
            return self.q.get(timeout=1)
        except queue.Empty:
            return None


def camera_streamin():
    try:
        start_time = time.time()
        camera_link = r"C:\Users\amarn\Downloads\istockphoto-1495533648-640_adpp_is.mp4"
        # camera_link = 0
        cap = Videocapture(camera_link)
        display_time = 1
        fc = 0
        person_count = 0

        while True:
            TIME = time.time() - start_time
            if TIME >= display_time:
                FPS = fc / TIME
                fc = 0
                start_time = time.time()
                FPS = round(FPS)
                fps_disp = "FPS: " + str(FPS)
                print(fps_disp)  # Print FPS for debugging

            frame = cap.read()
            if frame is not None:
                frame = cv2.resize(frame, (640, 480))
                

                results = model(frame)
                
                if results and len(results[0].boxes) > 0:
                    
                    for d in results[0].boxes:
                        
                        cls = int(d.cls.item())  # Convert to integer
                        
                        conf = d.conf.item()  # Convert to float
                        x1, y1, x2, y2 = map(int, d.xyxy[0])

                        label_name = CLASS_NAMES.get(cls, "unknown")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label_name} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        if label_name == "drone":
                            person_count += 1

                cv2.imshow("frame", frame)
                fc += 1

                if cv2.waitKey(1) == ord("q"):
                    break

        # cap.cap.release()
        cv2.destroyAllWindows()
    except Exception as error:
        logger.error(f"camera_streamin error : {error}")


camera_streamin()
def Streams_Parallel():
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        results = [executor.submit(camera_streamin)]
        
        for _ in concurrent.futures.as_completed(results):
            # future.result()  # Handle any exceptions
            pass

if __name__ == "__main__":
    Streams_Parallel()

