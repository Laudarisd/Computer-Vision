import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time



gpu = 0

# set the modified tf session as backend in keras

setup_gpu(gpu)



model_path = os.path.join('./snapshots/resnet50_csv_1000.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'vita_500', 1: 'gas_hwal', 2: 'hongsam', 3: 'dailyC', 4: 'mango', 5: 'red_bull', 6: 'gal_bae', 7: 'tejava', 8: 'power', 9: 'peach', 10: 'sol', 11: 'grape', 12: 'pocari', 13: '2%'}





# load image
image = read_image_bgr('./1.jpg')
output_path = './1_result.jpg'

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

def draw_box(image, box, color, thickness=5):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)



def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    #cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)


def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break
        
    color = label_color(label)
    
    b = box.astype(int)
    draw_box(draw, b, color=color)
    
    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)
    detected_img =cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, detected_img)
    
#plt.figure(figsize=(15, 15))
#plt.axis('off')
#cv2.imwrite(draw)
#plt.show()

"""
def round_int(x):
    if x == float("inf") or x == float("-inf"):
        return float('nan') # or x or return whatever makes sense
    return int(round(x))

def detect(net, meta, image, model, train_num, thresh=.3, hier_thresh=.7, nms=.15):
    im = load_image("./test_img/%s"%image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
    pont = ImageFont.truetype(fontpath, 70)
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    img = cv2.imread("./test_img/%s"%image)
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))

                if model == "distance":
                    cv2.rectangle(img, (int(b.x-(b.w/2)), int(b.y-(b.h/2))), (int(b.x+(b.w/2)), int(b.y+(b.h/2))), (255, 0, 0), 3)
                elif model == "no_distance":
                    cv2.rectangle(img, (int(b.x-(b.w/2)), int(b.y-(b.h/2))), (int(b.x+(b.w/2)), int(b.y+(b.h/2))), (0, 255, 0), 3)
                elif model == "nobrand":
                    cv2.rectangle(img, (int(b.x-(b.w/2)), int(b.y-(b.h/2))), (int(b.x+(b.w/2)), int(b.y+(b.h/2))), (0, 0, 255), 3)
                elif model == "total":
                    cv2.rectangle(img, (int(b.x-(b.w/2)), int(b.y-(b.h/2))), (int(b.x+(b.w/2)), int(b.y+(b.h/2))), (255, 255, 255), 3)
                elif model == "classification":
                    cv2.rectangle(img, (int(b.x-(b.w/2)), int(b.y-(b.h/2))), (int(b.x+(b.w/2)), int(b.y+(b.h/2))), (0, 0, 255), 3)
                    # cv2.putText(img, str(meta.names[i]), (int(b.x), int(b.y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    # cv2.putText(img, dic[str(meta.names[i])], (int(b.x), int(b.y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    # cv2.putText(img, dic[str(meta.names[i])], (int(b.x), int(b.y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    img_pil = Image.fromarray(img)
                    draw = ImageDraw.Draw(img_pil)
                    bb,g,r,a = 255,255,255,0
                    ff_t = str(str(meta.names[i]))
                    ff_t = ff_t.decode('utf-8')
                    draw.text((int(b.x-(b.w/2)), int(b.y-(b.h/2-100))), ff_t,font=pont, fill=(bb,g,r))
                    print(str(meta.names[i]))
                    img = np.array(img_pil)
                    
"""

