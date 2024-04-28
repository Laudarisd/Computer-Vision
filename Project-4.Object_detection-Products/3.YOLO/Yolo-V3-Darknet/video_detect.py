from ctypes import *
import os
import math
import glob
import random
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
#import tkinter
import csv
import time
import copy



darknet_so_path = "/home/sudip/YOLO_peru/libdarknet.so"
yolo_cfg_path = "/home/sudip/YOLO_peru/custom/yolov3.cfg"
yolo_weight_path = "/home/sudip/YOLO_peru/model/yolov3_100000.weights"
yolo_meta_path = "/home/sudip/YOLO_peru/custom/trainer.data"
#font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
#font_path = cv2.FONT_HERSHEY_PLAIN

#CLASSES = ["gal_bae", "dailyC", "mango", "peach", "gas_hwal", "grape", "hongsam", "vita_500", "pocari", "power", "tejava", "red_bull", "2%", "sol"]



def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

#COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def cvDrawBoxes(detections, img, original_frame):
    name_list = []
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 3)
        #cv2.rectangle(img, pt1, pt2, COLORS[idx], 3)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    [255,0 , 0], 3)

    return img, name_list


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



lib = CDLL(darknet_so_path, RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.7):
    """if isinstance(image, bytes):
        # image is a filename
        # i.e. image = b'/darknet/data/dog.jpg'
        im = load_image(image, 0, 0)
    else:
        # image is an nparray
        # i.e. image = cv2.imread('/darknet/data/dog.jpg')
        im, image = array_to_image(image)
        rgbgr_image(im)
    """
    im, image = array_to_image(image)
    rgbgr_image(im)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i],
                           (b.x, b.y, b.w, b.h)))
                #print(str(meta.names[i]))

    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes): free_image(im)
    free_detections(dets, num)
    return res




if __name__ == "__main__":    

    # load model
    net = load_net(yolo_cfg_path, yolo_weight_path, 0)
    meta = load_meta(yolo_meta_path)

    cap = cv2.VideoCapture(-1)
    
    ####################################################################################################
    # video option
    YUV2_CODEC = 844715353.0 # YUY2
    MJPG_CODEC = 1196444237.0 # MJPG
    cap_FPS = 15
    cap_AUTOFOCUS = 0
    cap_FOCUS = 0
    cap.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
    cap.set(3, 3264)
    cap.set(4, 2448)
    cap.set(cv2.CAP_PROP_FPS, cap_FPS)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, cap_AUTOFOCUS)
    cap.set(cv2.CAP_PROP_FOCUS, cap_FOCUS)
    
    #fontpath = font_path
    #pont = ImageFont.truetype(fontpath, 70)
    pont = cv2.FONT_HERSHEY_PLAIN
    ####################################################################################################

    while True:
        name_list = []
        prev_time = time.time()
        ret, frame = cap.read()
        original_frame = frame
        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        r = detect(net, meta, frame)
        img, name_list = cvDrawBoxes(r, frame, original_frame)
        
        k = cv2.waitKey(1) & 0xFF

        img = cv2.resize(img, (1280, 720))
        cv2.imshow('test', img)
        #os.system("clear")
        #print("FPS : " + str(1/(time.time()-prev_time)))

    cap.release()
