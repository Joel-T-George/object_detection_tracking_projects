import cv2
import torch ,os 
import threading
import time
from queue import Queue
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

model = torch.hub.load('./yolov5','custom',path='./yolov5/runs/yolov5s_results2/weights/best.pt',source="local")
model.to('cpu').eval()

cap = cv2.VideoCapture("/home/dhaksha/Desktop/detection-yolo/19-august-24-model-9.mp4", cv2.CAP_GSTREAMER)

# Define the directory to save frames
save_dir = "captured_frames"
os.makedirs(save_dir, exist_ok=True)

frame_count = 0
prev_frame = 0
new_frame =0
fps =0
font =cv2.FONT_HERSHEY_SIMPLEX

frame_queue =Queue(maxsize=5)
result_queue = Queue(maxsize=5)

def capture_frames():
    while True:
        ret , frame = cap.read()
        if not ret :
            break
        if not frame_queue.full():
            frame_queue.put(frame)

        time.sleep(0.01)
    cap.release()

def detection_interface():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results =model(frame)
            if not result_queue.full():
                result_queue.put((frame,results))

def display_results():
    prev_frame = time.time()
    try:
        while True:
            if not result_queue.empty():
                frame, detection = result_queue.get()
                boxes = detection.xyxy[0].cpu().numpy()
                labels = detection.names
                for box in boxes:
                    x1 ,y1,x2,y2 ,conf,cls= box 
                    cropped_img = frame[int(y1):int(y2),int(x1):int(x2)]
                    #cv2.imwrite(save_dir+f"/Cropped_image_{frame_count}.jpg",cropped_img)
                    cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                    label = f"{labels[int(cls)]} {conf:.2f}"
                    cv2.putText(frame,label,(int(x1),int(y1)-10),font,0.5,(0,255,0),2)
                    #frame_count = frame_count+1  
                
                new_frame = time.time()  
                fps = 1/(new_frame-prev_frame)
                prev_frame = new_frame
                fps = int(fps)
                cv2.putText(frame, f"fps: {fps}", (10,70), font, 1,(0,255,255),2) 
                cv2.imshow('Yolov5 Interface Multi-Threading App', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            
    except KeyboardInterrupt:
        print("\nProcess interface by the user")
    finally:
        cv2.destroyAllWindows()
try:
    capture_thread = threading.Thread(target=capture_frames)
    inference_thread = threading.Thread(target=detection_interface)
    display_thread = threading.Thread(target=display_results)

    capture_thread.start()
    inference_thread.start()
    display_thread.start()

    capture_thread.join()
    inference_thread.join()
    display_thread.join()  
     
except KeyboardInterrupt:
    print("User Killed Processing")