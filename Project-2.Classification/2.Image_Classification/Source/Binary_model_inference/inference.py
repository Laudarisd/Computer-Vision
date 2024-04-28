import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from efficientnet.tfkeras import preprocess_input
from PIL import Image
import xml.etree.ElementTree as ET
from xml.dom import minidom
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import os
import sys
import imutils
import time
from tensorflow.keras.models import load_model


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


def crop(image, box):
    xmin, ymin, xmax, ymax = box
    result = image[ymin:ymax+1, xmin:xmax+1, :]
    return result


def crop_image(image, boxes, resize=None, save_path=None):
    # image: cv2 image
    images = list(map(lambda b : crop(image, b), boxes)) 
    # boxes: [[xmin, ymin, xmax, ymax], ...] <- 이걸로 crop

    if str(type(resize)) == "<class 'tuple'>":
        try:
            images = list(map(lambda i: cv2.resize(i, dsize=resize, interpolation=cv2.INTER_LINEAR), images))
        except Exception as e:
            print(str(e))
    return images


def get_boxes(label_path):
    label_path = label_path
    xml_list = os.listdir(label_path)

    boxes_1 = {}
    cnt = 0
    for xml_file in sorted(xml_list):
        if xml_file =='.DS_Store':
            pass
        else:
            xml_path = os.path.join(label_path, xml_file)

            root_1 = minidom.parse(xml_path)  # xml.dom.minidom.parse(xml_path)
            bnd_1 = root_1.getElementsByTagName('bndbox')

            result = []
            for i in range(len(bnd_1)):
                xmin = int(bnd_1[i].childNodes[1].childNodes[0].nodeValue)
                ymin = int(bnd_1[i].childNodes[3].childNodes[0].nodeValue)
                xmax = int(bnd_1[i].childNodes[5].childNodes[0].nodeValue)
                ymax = int(bnd_1[i].childNodes[7].childNodes[0].nodeValue)
                result.append((xmin,ymin,xmax,ymax))

            boxes_1[str(cnt)] = result
            cnt += 1
    
    return boxes_1


def inference(image, prev_image, results, stbs):
    imgs = crop_image(image, main_boxes, (224, 224))
    p_imgs = crop_image(prev_image, main_boxes, (224, 224))
    em_imgs = crop_image(image, em_boxes, (224, 224))
    
    for img, p_img, em_img, idx in zip(imgs, p_imgs, em_imgs, range(len(imgs))):
        #print(np.mean(img - p_img))
        if stbs[idx]:
            if np.mean(np.abs(img - p_img)) > 110:
                stbs[idx] = False
        else:
            if np.mean(np.abs(img - p_img)) < 110:
                main_result = CLASS_NAMES[predict(img, main_model)]
                empty_result = EM_CLASS_NAMES[predict(em_img, empty_model)]
                results[idx] = empty_result if empty_result == 'empty' else main_result
                stbs[idx] = True

    return results, stbs
        

def predict(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    predictions = model.predict(img, steps=1)
    score = np.argmax(predictions[0])
    return score


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
    except RuntimeError as e:
        print(e)


# load product label class
df = pd.read_csv('./labels_3.txt', sep = ' ', index_col=False, header=None)
CLASS_NAMES = df[0].tolist()
CLASS_NAMES = sorted(CLASS_NAMES)

# load binary empty label class
df = pd.read_csv('./labels_binary.txt', sep = ' ', index_col=False, header=None)
EM_CLASS_NAMES = df[0].tolist()
EM_CLASS_NAMES = sorted(EM_CLASS_NAMES)

print(CLASS_NAMES)
print(EM_CLASS_NAMES)

#model = tf.keras.models.load_model('./model/beverage.h5')
empty_model = tf.keras.models.load_model('./1.model/empty_4.h5')
main_model = tf.keras.models.load_model('./1.model/final_pog_list_cls_data_noise.h5')

# load boundary boxes
os.system('clear')
print("Please select a mode to inference. 'd' or 'w' ")

while True:
    m = input()

    if m == 'w':
        em_boxes = get_boxes('./4.xml_empty')['1']
        main_boxes = get_boxes('./4.xml')['1']
        break

    elif m == 'd':
        em_boxes = get_boxes('./4.xml_empty')['0']
        main_boxes = get_boxes('./4.xml')['0']
        break

    else:
        print("Press the button 'd' or 'w'.")
        continue

print(len(em_boxes))
print(len(main_boxes))

results = ['' for i in range(len(main_boxes))]
# print(len(results), result)
stbs = [False for n in range(len(main_boxes))]
# print(len(stbs, stbs))
prev_frame = None

##########################################################################
cap = cv2.VideoCapture(-1)  
if cap.isOpened() == False:
    print('카메라를 오픈 할 수 없습니다.')

frame_width = int(3840)
frame_height = int(2160)

MJPG_CODEC = 1196444237.0 # MJPG
cap_AUTOFOCUS = 0
cap_FOCUS = 0

cv2.namedWindow('Usb Cam', cv2.WINDOW_FREERATIO)
cv2.resizeWindow('Usb Cam', frame_height, frame_width)

cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
cap.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
cap.set(cv2.CAP_PROP_AUTOFOCUS, cap_AUTOFOCUS)
##########################################################################

while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (2592, 1944))
    # print(frame.shape)
    
    frame = imutils.rotate(frame, 0)
    
    if prev_frame is None:
        prev_frame = frame
        continue
        
    results, stbs = inference(frame, prev_frame, results, stbs)
    os.system('clear')

    if m == 'd':
        p_results = [i for i in reversed(results)]
        print(p_results)
    
    if m == 'w':
        print(results)
    

    for idx, ((xmin, ymin, xmax, ymax), res) in enumerate(zip(main_boxes, results)):
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
        frame = cv2.putText(frame, res, (xmin+5, ymin-90), 0, 1.25, (255, 0, 255), 3, cv2.LINE_AA)
    
    
    for idx, ((xmin, ymin, xmax, ymax), res) in enumerate(zip(em_boxes, results)):
    	frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

    # showed_frame = cv2.resize(frame, (1920, 1080))	
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

       
    prev_frame = frame

cap.release()
cv2.destroyAllWindows()
