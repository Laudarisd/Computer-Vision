#-*- coding: utf-8 -*-

"""
This code helps to collect raw images by opening web cam. 

keys to take snapshot

c == capture images
q == quit (close web cam)
s == save images
"""


author = "__ Sudip Laudari __"
email = " __email__"
date = " 2023/08/11"
license = " Required license"



import cv2
import numpy as np
import imutils
from datetime import datetime
import time



class CaptureSnaphot():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  
        #self.frame_width = int(cap.get(3))
        #self.frame_height = int(cap.get(4))
        #self.fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        #self.out = cv2.VideoWriter('output.avi', fourcc, 25.0, (frame_width,frame_height))
        self.frame_width = int(3840)
        self.frame_height = int(2160)
        #self.frame_height = int(3840)
        #self.frame_width = int(2160)
        self.MJPG_CODEC = 1196444237.0 # MJPG
        self.cap_AUTOFOCUS = 0
        self.cap_FOCUS = 200
        #self.cap_ZOOM = 400
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
        self.cap.set(cv2.CAP_PROP_FOURCC, self.MJPG_CODEC)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, self.cap_AUTOFOCUS)
        self.cap.set(cv2.CAP_PROP_FOCUS, self.cap_FOCUS)
        #self.cap.set(cv2.CAP_PROP_ZOOM, cap_ZOOM)
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    def Window(self): 
        cv2.namedWindow('Usb Cam', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Usb Cam', self.frame_width, self.frame_height)
        if self.cap.isOpened() == False:
            print('Can not open the camera')
    #print cap.get(cv2.CAP_PROP_BRIGHTNESS)
    #print cap.get(cv2.CAP_PROP_FOURCC)
    def Rotate(self, src, degrees) :
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
    def Capture(self):
        while True:	
            ret, frame = self.cap.read()
        #change to frame size 
        #frame = cv2.resize(frame, (frame_width, frame_height))
        #frame = Rotate(frame, 180)
            frame = imutils.rotate(frame, 0)
            cv2.imshow('Usb Cam', frame)
                #out.write(frame)
            today = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
            ch = cv2.waitKey(1)
            if ch == ord('q'):
                break
            elif ch == ord('c'):
                print('press c')
                cv2.imwrite('./dataset/{subfolder_name}/usbcam({}).jpg'.format(today),frame)
                print('./dataset/{subfolder_name}/usbcam({}).jpg saved'.format(today))
            elif ch == ord('s'):
                for i in range(2):
                    cv2.imwrite('./dataset/{subfolder_name}/usbcam({})_{}.jpg'.format(today,i),frame)
                    print('./dataset/{subfolder_name}/usbcam({})_{}.jpg saved'.format(today,i))
                    time.sleep(1)
        self.cap.release()
        cv2.destroyAllWindows()
    def run(self):
        self.Window()
        self.Capture()
capture_img = CaptureSnaphot()
print(capture_img.run())
