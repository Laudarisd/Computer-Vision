# Object detection with YOLO-V4

This work is based on (https://github.com/AlexeyAB/darknet).



## Install darknet frame work 
```
$ git clone https://github.com/AlexeyAB/darknet.git
```

###### Go to ` darknet ` directory

```
cd darknet
```

- edit :
```
GPU=
CUDNN=
CUDNN_HALF=
....
zed_cam=

```

- run make 
```
$ make

```
------------------------------------------------------------------
## Make a folder name `custom` in darknet home 

```
$ darknet/ mkdir custom

```

-------------------------------------------------------------


## To train our own datasets we need to prepare following files inside of `custom` folders.



- Make `images` folder inside of custom folder and put all images here

    * After collecting all the images run `$ python3 split.py` which will create two files with name `test.txt` and `train.txt` inside of custom folder

    ```
    eg.
     
    #test.txt
    ./custom/images/usbcam(2020-03-13-10:26:43).jpg

    # train.txt

    ./custom/images/usbcam(2020-03-18-17:13:19).jpg


    ```

- Make `annotations` folder inside of custom folderand put all annotation files in this folder (if anotation files are in `.xml` format)


    * `.xml` files need to change in yolo format i.e `.txt` format. To chnage all the `.xml` files to `,txt`, run `$ python3 convert_aml_to_txt.py` in custom folder directory (don't     forget to edit necessary lines in `convert_aml_to_txt.py` file
    * This will create `.txt` files inside of annotations folder. Move all `.txt` files inside of `images` folder
   
    ```
    # .txt file will look like this:
    
    0 0.111520 0.499183 0.222426 0.997549
    1 0.393842 0.563725 0.227022 0.857843
    2 0.626379 0.538603 0.144301 0.609069
    
    ```
        
- Make `obj.data`
    * file which contains:
       
       ```
        classes=3                   # write number of classes
        train=./custom/train.txt    # path to train.txt
        valid=./custom/test.txt     # path to test.txt
        names=./custom/objects.names   
        backup=./backup


        ``` 




- object.names
    * with objects names - each in new line.

```
        1.aloe
        2.apple
        3.coca
```
 
- Edit `yolov4-custom.cfg` file (this file can be found inside of `cfg` folder)

    * change line batch to [`batch=64`]
    * change line subdivisions to [`subdivisions=16`]
    * change line max_batches to (`classes*2000` but not less than number of training images. For more information check: (https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L20)
    * change line steps to 80% and 90% of max_batches
    * set network size `width=416 height=416` or any value multiple of 32
    * change [`filters=255`] to filters=(classes + 5)x3 in the 3 `[convolutional]` before each `[yolo]` layers. If `classes=1` then should be `filters=18`. If `classes=2` then write `filters=21`
    
    * change line `classes=80` to your number of objects in each of 3 `[yolo]`-layers

---------------------------------------------------------------------------------------------
## Download darknet pre trained weight inside of `custom` folder

For training `cfg/yolov4-custom.cfg` download the pre-trained weights-file (162 MB): [yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137) (Google drive mirror [yolov4.conv.137](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp) ) 


------------------------------------------------------------------------------------------

## Taining data


```
./darknet detector train custom/obj.data custom/yolov4-custom.cfg custom/yolov4.conv.137
```

- To use gpu

```
./darknet detector train custom/obj.data custom/yolov4-custom.cfg custom/yolov4.conv.137 -gpus 0,1,2

```

-------------------------------------------------------------------------------------------


##  Test

```
./darknet detector test custom/obj.data custom/yolov4-custom.cfg backup/yolo-obj_last.weights test_img/1.jpg
 
``` 

## Webcam 

```
./darknet detector demo custom/obj.data custom/yolov3.cfg backup/yolo-obj_last.weights -c 0
```
---------------------------------------------------------------------------------------------------------------
                                          
 
 ## Main references
   * https://github.com/AlexeyAB/darknet




