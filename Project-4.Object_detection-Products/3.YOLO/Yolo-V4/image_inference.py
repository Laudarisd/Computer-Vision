from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue

weight_file = "/data/backup/darknet/weights/cigar_box_2020.12.30/yolov4-custom_cigar_box_last.weights"
config_file = "/data/backup/darknet/build/darknet/x64/cfg/yolov4-custom_cigar_box.cfg"
data_file = "/data/backup/darknet/build/darknet/x64/data/cigar_box_obj.data"
thresh_hold = .6

network, class_names, class_colors = darknet.load_network(config_file, data_file, weight_file, batch_size=1)

width = darknet.network_width(network)
height = darknet.network_height(network)
darknet_image = darknet.make_image(width, height, 3)


test_image = cv2.imread("/data/backup/darknet/build/darknet/x64/data/obj/cigarette_case/1.seed/backup_images/part2_1217_nm/2020-12-17-09:48:03_part2_1217_nm_left.jpg")
frame_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh_hold)
# darknet.print_detections(detections)

res = []
for i in range(len(detections)):
    res.append(detections[i][0])

image = darknet.draw_boxes(detections, frame_resized, class_colors)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite("/data/backup/darknet/build/darknet/x64/results/predictions.jpg", image)

print(res)
cv2.imshow("inference", image)
k = cv2.waitKey(0)
