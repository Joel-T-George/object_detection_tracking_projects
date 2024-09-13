def send_packet(socket,packet,GIMBAL_INFO:tuple):
    socket.sendto(bytearray.fromhex(packet), GIMBAL_INFO)


rotate_left = "55 66 01 02 00 00 00 07 64 64 3d cf" 
rotate_right = "55 66 01 02 00 00 00 07 FF FF xx yy"
rotate_up = "55 66 01 02 00 00 00 07 64 64 xx yy"
rotate_down = "55 66 01 02 00 00 00 07 64 64 xx yy"

def track_object_to_center(objectX, objectY, image_width, image_height):
    centerX = image_width // 2
    centerY = image_height // 2
    deltaX = objectX - centerX
    deltaY = objectY - centerY

    # Adjust gimbal pitch and yaw based on deltas
    if abs(deltaX) > 10:  # Threshold for movement (can be adjusted)
        if deltaX > 0:
            rotate_right()  # Rotate gimbal right
        else:
            rotate_left()  # Rotate gimbal left

    if abs(deltaY) > 10:
        if deltaY > 0:
            rotate_down()  # Rotate gimbal down
        else:
            rotate_up()  # Rotate gimbal up