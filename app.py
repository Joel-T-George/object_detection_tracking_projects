import cv2 
import numpy as np
import time
import Pid
import socket

ENABLE_FPS_RENDER =False

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('192.168.6.200', 10001)

controller1 = Pid.Pid(kp=0.5, ki=0.01, kd=0.1)
controller2 = Pid.Pid(kp=1.0, ki=0.01, kd=0.1)

point_selected =False
point = ()
p0 =None
old_gray = None
boundBox =None

def camera_tracking(center_x,center_y, controller1,controller2):
    
    control1 = controller1.compute1(center_x, 960)
    control2 = controller2.compute2(center_y, 540)
    pwm1 = control1+1500
    pwm2 = control2+1500
    pwm1 = round(pwm1)
    pwm2 = round(pwm2)
    roll = int(pwm1)     
    pitch = int(pwm2)
    res= str(roll) + "," + str(pitch)
    print("@@@",roll,pitch)
    if (roll < 1300 or roll > 1700) and (pitch < 1300 or pitch > 1700):
        res= str(roll) + "," + str(pitch)
        print("@@@",roll,pitch)         
        server_socket.sendto(str(res).encode(),server_address)
    elif (roll < 1300 or roll > 1700):
        res= str(roll) + "," + str(1500)
        print("@@@",roll,pitch) 
        server_socket.sendto(str(res).encode(),server_address)
    elif (pitch < 1300 or pitch > 1700):
        res= str(1500) + "," + str(pitch)
        print("@@@",roll,pitch) 
        server_socket.sendto(str(res).encode(),server_address)

#frames fixing
desired_fps = 60
frame_time =1/desired_fps

def select_point(event, x,y,flags,param):
    global point, point_selected, p0, boundBox
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x,y)
        point_selected = True
        p0 = np.array([[x,y]], dtype=np.float32)
        boundBox = (x-50, y-50,100,100)


cv2.namedWindow("Tracking Window")
cv2.setMouseCallback("Tracking Window", select_point)
pipline =f'rtspsrc location=rtsp://192.168.6.212:554/stream0 latency=0 ! decodebin ! videoconvert ! appsink'
cap = cv2.VideoCapture(pipline, cv2.CAP_GSTREAMER)

lk_params = dict(winSize = (15,15), maxLevel=2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))

while cap.isOpened():
    ret , frame = cap.read()
    if not ret:
        break
    cv2.circle(frame,(960,540),5, (0, 0, 255), -1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if point_selected:
        if old_gray is not None:
            p1, st,err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,p0,None,** lk_params)

            if st[0] ==1:
                x, y  = p1.ravel()
                p0 = p1

                x1, y1 = int(x-30), int(y-30)
                x2, y2 = int(x + 30), int(y + 30)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                roi = frame[y1:y2, x1:x2]
                h,w,_ = roi.shape
                if h > 0 and w>0 :
                    cv2.imshow("Region of Interset",roi)
                    
                camera_tracking(x,y,controller1,controller2)

        old_gray = frame_gray.copy()
    
    cv2.imshow("Tracking Window", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

