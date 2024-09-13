import time, socket,sys


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('192.168.6.200', 10001)

sock.bind(server_address)
sock.setblocking(0)

# Left = [0xEB ,0x90 ,0x14 ,0x55 ,0xAA ,0xDC ,0x11 ,0x30 ,0x01 ,0xF8 ,0x30 ,0x00 ,0x00 ,0x00 ,0x00 ,0x00 ,0x00 ,0x00 ,0x00 ,0x00 ,0x00 ,0x00 ,0xE8 ,0x2D]
# right = "EB 90 14 55 AA DC 11 30 01 07 D0 00 00 00 00 00 00 00 00 00 00 00 F7 EB"
# up = "EB 90 14 55 AA DC 11 30 01 00 00 07 D0 00 00 00 00 00 00 00 00 00 F7 EB"
# down = "EB 90 14 55 AA DC 11 30 01 00 00 F8 30 00 00 00 00 00 00 00 00 00 E8 2D"
# stop = "EB 90 14 55 AA DC 11 30 01 00 00 00 00 00 00 00 00 00 00 00 00 00 20 3D"

Left = "EB 90 14 55 AA DC 11 30 01 F8 30 00 00 00 00 00 00 00 00 00 00 00 E8 2D"
right = "EB 90 14 55 AA DC 11 30 01 07 D0 00 00 00 00 00 00 00 00 00 00 00 F7 EB"
up = "EB 90 14 55 AA DC 11 30 01 00 00 07 D0 00 00 00 00 00 00 00 00 00 F7 EB"
down = "EB 90 14 55 AA DC 11 30 01 00 00 F8 30 00 00 00 00 00 00 00 00 00 E8 2D"
stop = "EB 90 14 55 AA DC 11 30 01 00 00 00 00 00 00 00 00 00 00 00 00 00 20 3D"


zoom_plus = [0x81,0x01,0x04,0x07,0x27,0xFF]

zoom_minus = [0x81,0x01,0x04,0x07,0x37,0xFF]

zoom_stop = [0x81,0x01,0x04,0x07,0x00,0xFF]

prev_pitch, prev_yaw, prev_zoom = 0,0,0

s6 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

HOST = '192.168.6.212'  # The server's hostname or IP address
PORT = 2000  # The port used by the server

def is_connected(sock):
    try:
        sock.send(b'')
    except socket.error:
        return False
    return True

def connect_to_gimbal():
    try:
        s6.connect((HOST, PORT))
        time.sleep(2)
        print("Connected to gimbal")
    except socket.error as e:
        print(f"Connection failed: {e}")
        sys.exit(1)  # Exit the program if the connection fails

connect_to_gimbal()

while True:
    try:
        data,address = sock.recvfrom(1024)
        #print(data.decode())
        res = data.decode()
        res=res.split(",")
        print("lengthhhhhhhhhhhh",len(res))
        if len(res) == 2:
            yaw , pitch, zoom, zoom_out = int(float(res[0])), int(float(res[1])), 50, 50
            print(yaw,pitch,zoom,zoom_out)
        else:
            yaw , pitch, zoom, zoom_out = int(float(res[0])), int(float(res[1])), int(float(res[2])), int(float(res[3]))
            print(yaw,pitch,zoom,zoom_out)
        print(yaw,pitch,zoom,zoom_out)

        if pitch > 1300 and pitch < 1700 and pitch != prev_pitch:
            packets =bytes.fromhex(stop)
            s6.sendall(packets)
            #time.sleep(0.5)
            print("stoppppppppppppp Pitchhhhhh")
        elif pitch > 1600 and pitch != (prev_pitch+5):
            packets =bytes.fromhex(up)
            s6.sendall(packets)
            time.sleep(0.5)
            print("upppppppppppppppppp Pitchhhhhhhhh")
        elif pitch < 1400 and pitch != (prev_pitch-5):
            packets =bytes.fromhex(down)
            s6.sendall(packets)
            #time.sleep(0.5)
            print("downnnnnnnnnnnnnnnnnn Pitchhhhhhhhh")
            
        if yaw > 1300 and yaw < 1700 and yaw != prev_yaw:
            packets =bytes.fromhex(stop)
            s6.sendall(packets)
            #time.sleep(0.5)
            print("stoppppppppppppp yawwwwwww")
        elif yaw > 1600 and yaw != (prev_yaw+5):
            packets =bytes.fromhex(Left)
            s6.sendall(packets)
            #time.sleep(0.5)
            print("lefttttttttttt yawwwwwwwwwww")
        elif yaw < 1400 and yaw != (prev_yaw-5):
            packets =bytes.fromhex(right)
            s6.sendall(packets)
            #time.sleep(0.5)
            print("righttttttttttttttt yawwwwwwwwwwwwww")
            
        if zoom > 40 and zoom <60 and zoom != prev_zoom:
            packets =bytes.fromhex(stop)
            s6.sendall(zoom_stop)
            #time.sleep(0.5)
            print("stoppppppppppppp zoommmmmmmmmmm")
        elif zoom > 60 and zoom != (prev_zoom+5):
            packets =bytes.fromhex(stop)
            s6.sendall(zoom_plus)
            #time.sleep(0.5)
            print("plussssssssssss zoommmmmmmmmmmm")
        elif zoom < 40 and zoom != (prev_zoom-5):
            packets =bytes.fromhex(stop)
            s6.sendall(zoom_minus)
            #time.sleep(0.5)
            print("minusssssssssssss zoommmmmmmmmmmmmmm")
        prev_pitch,prev_yaw,prev_zoom = pitch,yaw,zoom
    except:
        pass