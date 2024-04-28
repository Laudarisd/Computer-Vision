# Keras RetinaNet-Object Detection

## Object detection on custom dataset

Keras implimentation of Retinanet object detection as described in <https://github.com/fizyr/keras-retinanet>

---------------------------------------------------------------------------------------------------

## Installation

1) Clone this repository or 

clone this: 
```
https://github.com/fizyr/keras-retinanet

```
--------------------------------------------------------------------------------------------
## Requirements: 

- numpy
- python-3
- tensorflow 
- keras
--------------------------------------------------------------------------------------------
## Setup-
1) In the repository, execute `pip install . --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
2) Alternatively, you can run the code directly from the cloned  repository, however you need to run `python setup.py build_ext --inplace` to compile Cython code first.
---------------------------------------------------------------------------------------------
## Data preprocessing
1) Collect images and annotatiion files corresponding to images. 
2) Keep them in images and annotations folders.


We need to change our data to keras-retina format so that we can train them. To change our data follow the following steps: 

### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
path/to/image.jpg,x1,y1,x2,y2,class_name
```
By default the CSV generator will look for images relative to the directory of the annotations file.

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:
```
path/to/image.jpg,,,,,
```

A full example:
```
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,
```

This defines a dataset with 3 images.
`img_001.jpg` contains a mango.
`img_002.jpg` contains a mango and a peach.
`img_003.jpg` contains no interesting objects.

To convert our data in this format, first edit config - `object_detection_retinanet_config.py` file.
Then run 

```
$ python3 build_dataset.py 

```

This will create `test.csv`, `train.csv` and `classes.csv` files.

----------------------------------------------------------------------------------

## Pretrained models

All models can be downloaded from the [releases page](https://github.com/fizyr/keras-retinanet/releases).

-----------------------------------------------------------------------------------------
## Working directory:
![Derectory](../master/img_all/derectory.png)



------------------------------------------------------------------------------------------------
## Training
`keras-retinanet` can be trained using [this](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/bin/train.py) script.
Note that the train script uses relative imports since it is inside the `keras_retinanet` package.
If you want to adjust the script for your own use outside of this repository,
you will need to switch it to use absolute imports.

If you installed `keras-retinanet` correctly, the train script will be installed as `retinanet-train`.
However, if you make local modifications to the `keras-retinanet` repository, you should run the script directly from the repository.
That will ensure that your local changes will be used by the train script.

The default backbone is `resnet50`. You can change this using the `--backbone=xxx` argument in the running script.
`xxx` can be one of the backbones in resnet models (`resnet50`, `resnet101`, `resnet152`), mobilenet models (`mobilenet128_1.0`, `mobilenet128_0.75`, `mobilenet160_1.0`, etc), densenet models or vgg models. The different options are defined by each model in their corresponding python scripts (`resnet.py`, `mobilenet.py`, etc).

Trained models can't be used directly for inference. To convert a trained model to an inference model, check [here](https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model).



For training on a [custom dataset], a CSV file can be used as a way to pass the data.
See below for more details on the format of these CSV files.
To train using your CSV, run:

```
keras_retinanet/bin/train.py csv /path/to/csv/file/containing/annotations /path/to/csv/file/containing/classes
```


or 

```
retinanet-train --backbone resnet101 --weights resnet50_coco_best_v2.1.0.h5 --steps 400 --epochs 100 --snapshot-path snapshots --tensorboard-dir tensorboard csv train.csv classes.csv
```


-----------------------------------------------------------------------------------------------------

## Testing
Run following script for single image (This code includes model convertion steps).

```
$ python3 test.py
```

## Results

Example output images using `keras-retinanet` are shown below.

<a href="url"><img src="https://github.com/Laudarisd/Keras-retinanet/blob/master/img_all/1_result.jpg"  alt="result_1" align="left" height="250" width="250" ></a>


<a href="url"><img src="https://github.com/Laudarisd/Keras-retinanet/blob/master/img_all/2_result.jpg" alt="result_2" align="left" height="250" width="250" ></a>

--------------------------------------------------------------------------------------------------------------------







# References 

<https://github.com/fizyr/keras-retinanet>




____________________________________________________________________________________________________________________
