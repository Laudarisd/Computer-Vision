# This is a PyTorch Tutorial to Object Detection in custom data set

----------------------------------------------------------------------------------------------
## Required :


- Basic knowledge of PyTorch, convolutional neural networks
- `PyTorch 0.4` in `Python 3.6`

------------------------------------------------------------------------------------------------------------------------------------------
Bsed on the above github repository, I tried to train my own data.


--------------------------------------------------------------------------------------------------------------------

## Pipeline 

## Need to create train.txt, test.txt and trainval.txt from our data.

Let's start from beginning

1. Collect images
2. Collect annotations


`labelimg` can be used to label all the images. It gives '.xml' which contains detail information about image path, difficulties, label map , coordinates and etc.

--------------------------------------------------------------------------------------------------------------------------------
**Check directory**
```
Pytorch_tutorial- data - dataset - images (all the images go here)
                - train.py       - Annotations (all the labels '.xml' file go here ) 
                - utils.py
                - model.py
                - ........
                - ........
                
```
--------------------------------------------------------------------------------------------------------------------------
## Convert raw data to pytorch format


- Go to `data` folder. Create `test.txt, train.txt and trainval.txt` by running `convert_txt.py` file.(Don't forget to edit necessary parts.e.g data path)
- This will save '.txt' file inside of datasets folder
```
$ python3 convert_txt.py

```
-------------------------------------------------------------------------------------------------------------------------------
## Create json file

- Copy `utils.py`, `create_data_lists.py` files from `Pytorch_tutorial` path and paste inside of `data` folder
- Edit `voc_labels` in `util.py` file and other necessary parts.Such as 'file path, 'file names' if required.
- Edit path in `create_data_lists.py`
- Put all `.txt` files inside of 'data' fiolder which were created before in 'dataset' folder
- Run `create_data_lists.py`


```
$ python3 create_data_lists.py

```

- This will create five json files in 'data'folder
```
                - TEST_images.json
                - TEST_labels.json
                - TRAIN_images.json
                - TRAIN_labels.json
                - label_map.json
                
 
```
            
            
- Move those files in `Pytorch_tutorial` folder.            
------------------------------------------------------------------------------------------------------------------------------------
# Training

The parameters for the model (and training it) are at the beginning of the file, If required, check or modify them.

To **train model from scratch**, run this file –

`$ python3 train.py`


------------------------------------------------------------------------------------------------------------------------------------------

# Evaluation

- Edit  label in `utils.py file

- Run `eval.py`

```
$ python3 eval.py

```

Class-wise average precisions (not scaled to 100) are listed below.

| Class | Average Precision |
| :-----: | :------: |
| _aqua_ | 1.0 |
| _coffee_ | 0.99 |
| _demi_ | 1.0 |
| _e_drink_ | 1.0 |
| _grape_soda_ | 1.0 |
| _juice_ | 0.90 |
| _lemon_ | 0.90 |
| _mango_ | 1.0 |
| _milkis_ | 1.0 |
| _peach_ | 1.0 |
| _pocari_ | 1.0 |
| _soda_ | 1.0|
| _sol_ | 1.0 |
| _vita_drink_ | 1.0 |
| _water | 0.98 |

--------------------------------------------------------------------------------------------------

# Inference

- Edit `detect.py` file
```python
img_path = '/path/to/ima.ge'
original_image = PIL.Image.open(img_path, mode='r')
original_image = original_image.convert('RGB')

detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
```

-Run ` detect.py`


```
$ python3 detect.py


```
----------------------------------------------------------------------------------------------------------

# References

<https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection>





----------------------------------------------------------------------------------------------------

 









