# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
import cv2
import random
import glob
from datetime import datetime
import shutil
import os
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomGamma, VerticalFlip,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, 
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, Rotate, RandomContrast, RandomBrightness, RandomCrop, Resize, OpticalDistortion
)

IMG_RESIZE = 512

def aug_options(p=1):
    return Compose([
        # Resize(IMG_RESIZE, IMG_RESIZE),

        # RandomCrop(IMG_RESIZE, IMG_RESIZE, p=0.5),  
        
        OneOf([
        RandomContrast(p=1, limit=(-0.5,1)),   # -0.5 ~ 2 -- RandomBrightnessContrast
        RandomBrightness(p=1, limit=(-0.2,0.1)),
        ], p=0.6),
            
        OneOf([
            Rotate(limit=(180, 180), p=0.3),
            RandomRotate90(p=0.3),
            VerticalFlip(p=0.3),
        ], p=0.5),

        OneOf([
            IAASharpen(alpha=(0.2, 0.5), lightness=(0.1, 0.5), always_apply=False, p=0.5),
            MotionBlur(p=0.1),
            HorizontalFlip(p=0.4)
        ], p=0.5),

        ShiftScaleRotate(shift_limit=0.001, scale_limit=0.1, rotate_limit=180, p=0.3, border_mode=1),
        Resize(IMG_RESIZE, IMG_RESIZE, p=1),
        ],
        p=p)


def apply_aug(aug, image):
    image = aug(image=image)['image']
    return image


def show_img_distribution(img_df):
    print(img_df)
    print(img_df['label'].value_counts().sort_index())
    print("Min : ", img_df['label'].value_counts().min())
    # plt.figure(figsize=(14, 6))
    img_df['label'].value_counts().sort_index().plot.barh(figsize=(14,10), title='Num of images Distribution')
    plt.xlabel('NUM OF IMAGES')
    plt.ylabel('CLASS_NAMES')
    plt.show()

    os.system("clear")


def show_splited_datasets(train_set, valid_set):
    labels = train_set['label'].sort_index()
    train_set_imgs = train_set['label'].value_counts().sort_index()
    valid_set_imgs = valid_set['label'].value_counts().sort_index()

    print("Train Set Distribution \n", train_set_imgs)
    print('=======================================================================')
    print("Validation Set Distribution \n", valid_set_imgs)

    index = []
    for l in labels:
        if l in index:
            pass
        else:
            index.append(l)

    train_imgs_num = []
    valid_imgs_num = []

    for i in train_set_imgs:
        train_imgs_num.append(i)

    for i in valid_set_imgs:
        valid_imgs_num.append(i)

    df = pd.DataFrame({'train imgs':train_imgs_num, 'valid imgs':valid_imgs_num}, index=index)

    df.plot.barh(title='Num of images Distribution', figsize=(14, 10))
    plt.show()

    os.system("clear")
    

def show_aug_sampels(path):
    print("Show augmented image samples")
    imgs = glob.glob(path + '/*/*.jpg')
    # print(len(imgs))
    idx = random.randint(0, len(imgs))
    aug = aug_options(p=1)

    for i in range(0, 30):
        image = cv2.imread(imgs[idx])
        image = cv2.resize(image, (IMG_RESIZE, IMG_RESIZE))
        aug_img = apply_aug(aug, image)

        # numpy_horizontal = np.hstack((image, aug_img))
        numpy_horizontal_concat = np.concatenate((image, aug_img), axis=1)
        # numpy_horizontal_concat = cv2.resize(numpy_horizontal_concat, (1280, 720))

        cv2.imshow('Original / Augmentation', numpy_horizontal_concat)
        cv2.waitKey(300)
    cv2.destroyAllWindows()

    os.system('clear')


def aug_processing(data_set, label_list, output_path, aug_num, is_train):
    img_path = data_set['image_path'].sort_index()
    labels = data_set['label'].value_counts().sort_index()

    # print(img_path)
    # print(labels)
    
    if is_train == True:
        output_path = output_path + '/train'

    else:
        output_path = output_path + '/valid'

    for path in img_path:
        file_name = path.split('/')[-1]
        label = path.split('/')[-2]
        avg = int(round(aug_num / labels[label]))

        aug = aug_options(p=1)
        image = cv2.imread(path)

        if not(os.path.isdir(output_path + '/' + label)):
            os.makedirs(output_path + '/' + label)

        else:
            pass
        
        num_of_imgs = len(glob.glob(output_path + '/' + label + '/*.jpg'))
        today = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

        for i in range(avg):
            aug_img = apply_aug(aug, image)
            cv2.imwrite(output_path + '/' + label + '/' + file_name + '_' + str(i) + '.jpg', aug_img)

            
    return output_path
                

def make_df(path):
    result = []
    idx = 0
    label_list = sorted(os.listdir(path))
    print(label_list)

    for label in label_list:
        file_list = glob.glob(os.path.join(path,label,'*'))
        
        for file in file_list:
            result.append([idx, label, file])
            idx += 1
            
    img_df = pd.DataFrame(result, columns=['idx','label','image_path'])

    return img_df


if __name__ == "__main__":
    # Dataset Path define
    path = '/data//datasets/test'
    dataset_name = path.split('/')[-1]
    output_path = '/data/Auged_datasets/' + dataset_name

    # Dataset Load & visualization
    result = []
    idx = 0
    label_list = sorted(os.listdir(path))
    print(label_list)

    for label in label_list:
        file_list = glob.glob(os.path.join(path,label,'*'))
        
        for file in file_list:
            result.append([idx, label, file])
            idx += 1
            
    img_df = pd.DataFrame(result, columns=['idx','label','image_path'])
    show_img_distribution(img_df)

    # Split Train set, Validation set + Visualization
    train_set, test_set = train_test_split(img_df, test_size=0.2, shuffle=True)
    show_splited_datasets(train_set, test_set)

    show_aug_sampels(path)

    while True:
        print("Start Aug Process??? Press y or n")
        a = input()
        

        if a == 'y':
            print('How many augmentation do you want?')
            aug_num = float(input())

            output_train = aug_processing(train_set, label_list, output_path, int(aug_num), is_train=True)
            output_train_df = make_df(output_train)

            output_valid = aug_processing(test_set, label_list, output_path, int(aug_num * 0.2), is_train=False)
            output_valid_df = make_df(output_valid)

            show_splited_datasets(output_train_df, output_valid_df)
            break


        elif a == 'n':
            break


        else:
            print("Please press y or n")
            continue
