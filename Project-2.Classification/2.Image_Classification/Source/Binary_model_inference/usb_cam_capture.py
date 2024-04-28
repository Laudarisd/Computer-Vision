#-*- coding: utf-8 -*-
import cv2
import numpy as np
import imutils
from datetime import datetime
import time

def Rotate(src, degrees) :
    if degrees == 90 :
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 1)
    elif degrees == 180 :
        dst = cv2.flip(src, 0)
    elif degrees == 270 :
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 0)
    else :
        dst = null
    return dst



cap = cv2.VideoCapture(0)  
if cap.isOpened() == False:
    print('카메라를 오픈 할 수 없습니다.')

#frame_width = int(cap.get(3))
#frame_height = int(cap.get(4))
#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#out = cv2.VideoWriter('output.avi', fourcc, 25.0, (frame_width,frame_height))

frame_width = int(3840)
frame_height = int(2160)

#frame_height = int(3840)
#frame_width = int(2160)



MJPG_CODEC = 1196444237.0 # MJPG

cap_AUTOFOCUS = 0

cap_FOCUS = 0
#cap_ZOOM = 400


cv2.namedWindow('Usb Cam', cv2.WINDOW_FREERATIO)
cv2.resizeWindow('Usb Cam', frame_height, frame_width)

cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
cap.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

cap.set(cv2.CAP_PROP_AUTOFOCUS, cap_AUTOFOCUS)
cap.set(cv2.CAP_PROP_FOCUS, cap_FOCUS)

#cap.set(cv2.CAP_PROP_ZOOM, cap_ZOOM)


#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

#print cap.get(cv2.CAP_PROP_BRIGHTNESS)
#print cap.get(cv2.CAP_PROP_FOURCC)


while True:	

    ret, frame = cap.read()

#change to frame size 
#frame = cv2.resize(frame, (frame_width, frame_height))
#frame = Rotate(frame, 180)

    frame = imutils.rotate(frame, 0)
    # frame_sh = cv2.resize(frame, (1920,1080))
    cv2.imshow('Usb Cam', frame)
        #out.write(frame)
    today = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    ch = cv2.waitKey(1)
    if ch == ord('q'):
	    break

    elif ch == ord('c'):
        print('press c')
        cv2.imwrite('./saved_images/usbcam({}).jpg'.format(today),frame)
        print('./saved_images/usbcam({}).jpg saved'.format(today))


    elif ch == ord('s'):
        for i in range(2):
            cv2.imwrite('./saved_images/usbcam({})_{}.jpg'.format(today,i),frame)
            print('./saved_images/usbcam({})_{}.jpg saved'.format(today,i))
            time.sleep(1)


cap.release()
cv2.destroyAllWindows()
