
import cv2
import os
import torch
import time
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)



#cap = cv2.VideoCapture("rtspsrc location=rtsp://192.168.6.126:554/stream0 latency=0 ! decodebin ! videoconvert ! appsink", cv2.CAP_GSTREAMER)


cap = cv2.VideoCapture("/home/dhaksha/Desktop/detection-yolo/19-august-24-model-9.mp4", cv2.CAP_GSTREAMER)
model = torch.hub.load('./yolov5','custom',path='./yolov5/runs/yolov5s_results2/weights/best.pt',source="local")
model.to('cuda').eval()

# Define the directory to save frames
save_dir = "captured_frames"
os.makedirs(save_dir, exist_ok=True)

# Initialize a frame counter
frame_count = 0
prev_frame = 0
new_frame =0
fps =0
font =cv2.FONT_HERSHEY_SIMPLEX
try:
    if not cap.isOpened():
        print("Error: Unable to open the webcam")
    else:
        while True:
            ret, frame = cap.read()
            


            if not ret:
                print("Failed to grab frame")
                break

            # Perform object detection on the frame
            results = model(frame)

            boxes = results.xyxy[0].cpu().numpy()
            labels =results.names
            for box in boxes:
                x1 ,y1,x2,y2 ,conf,cls= box 
                cropped_img = frame[int(y1):int(y2),int(x1):int(x2)]
                cv2.imwrite(save_dir+f"/Cropped_image_{frame_count}.jpg",cropped_img)
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                label = f"{labels[int(cls)]} {conf:.2f}"
                cv2.putText(frame,label,(int(x1),int(y1)-10),font,0.5,(0,255,0),2)
                
                frame_count = frame_count+1
            #print(results.xyxy)
            new_frame = time.time()
            fps = 1/(new_frame - prev_frame)
            prev_frame = new_frame
            fps =int(fps)
            cv2.putText(frame, f"fps: {fps}", (10,70), font, 1,(0,255,255),2)
            
            # img_with_boxes = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR)
            cv2.imshow('YOLOv5 Interface', frame)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
except:
    print("\nProcess interrupted by the user.")

finally:
    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

