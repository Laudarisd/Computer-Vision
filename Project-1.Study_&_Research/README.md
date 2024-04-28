Table of contents
=================

<!--ts-->
   * [Overview ](https://github.com/Laudarisd/Deep-learning-and-ML-preparation)
   * [Object_detection](https://github.com/Laudarisd/Deep-learning-and-ML-preparation/tree/main/src/object_detection)
   * [image_classification](https://github.com/Laudarisd/Deep-learning-and-ML-preparation/tree/main/src/image_classification)
   * [Machine_Learning_all](https://github.com/Laudarisd/Deep-learning-and-ML-preparation/tree/main/src)
      * [1-Data_Preprocessing](https://github.com/Laudarisd/Deep-learning-and-ML-preparation/tree/main/src/Machine_Learning_all)
      * [2-Regression](https://github.com/Laudarisd/Deep-learning-and-ML-preparation/tree/main/src/Machine_Learning_all)
      * [3-Classification](https://github.com/Laudarisd/Deep-learning-and-ML-preparation/tree/main/src/Machine_Learning_all)
      * [4-Clustering](https://github.com/Laudarisd/Deep-learning-and-ML-preparation/tree/main/src/Machine_Learning_all)
      * [5-Association_Rule_Learning](https://github.com/Laudarisd/Deep-learning-and-ML-preparation/tree/main/src/Machine_Learning_all)
      * [6-Reinforcement_Learning](https://github.com/Laudarisd/Deep-learning-and-ML-preparation/tree/main/src/Machine_Learning_all)
      * [7-Natural_Language](https://github.com/Laudarisd/Deep-learning-and-ML-preparation/tree/main/src/Machine_Learning_all)
      * [8-Deep_Learning](https://github.com/Laudarisd/Deep-learning-and-ML-preparation/tree/main/src/Machine_Learning_all)
      * [9-Dimensionality_Reduction](https://github.com/Laudarisd/Deep-learning-and-ML-preparation/tree/main/src/Machine_Learning_all)
      * [10-Model_Selection_Boosting](https://github.com/Laudarisd/Deep-learning-and-ML-preparation/tree/main/src/Machine_Learning_all)
   * [Other_problems](https://github.com/Laudarisd/Deep-learning-and-ML-preparation/tree/main/src/Other_problems)
   * [Numerical_problems](https://github.com/Laudarisd/Deep-learning-and-ML-preparation/tree/main/src/numerical_problems)
<!--te-->



Overview Image Classification & Object Detection
============================================================

**[Image Classification](https://github.com/Laudarisd/Interview_exam_preparation/tree/main/src/image_classification)**
=======================================================================================================================

In simple words, image classification is a technique that is used to classify or predict the class of a specific object in an image. The main goal of this technique is to accurately identify the features in an image.

**How Image Classification Works**
In general, the image classification techniques can be categorised as parametric and non-parametric or supervised and unsupervised as well as hard and soft classifiers. For supervised classification, this technique delivers results based on the decision boundary created, which mostly rely on the input and output provided while training the model. But, in the case of unsupervised classification, the technique provides the result based on the analysis of the input dataset own its own; features are not directly fed to the models.

<table border="0">
   <tr>
      <td>
      <img src="./src/img/cl1.png" width="100%" />
      </td>
      <td>
      <img src="./src/img/cl2.png" width="100%" />
      </td>
   </tr>
   </table>

[image-source](https://www.google.com/search?q=image+classification&tbm=isch&ved=2ahUKEwjLkbHE_JzsAhUMBpQKHbvuAvAQ2-cCegQIABAA&oq=image&gs_lcp=CgNpbWcQARgAMgQIABBDMgIIADIECAAQQzIECAAQQzIECAAQQzIECAAQQzIFCAAQsQMyBAgAEEMyBAgAEEMyBAgAEEM6BwgAELEDEENQwfgCWK-JA2DPmQNoAHAAeAOAAXOIAfcOkgEENC4xNJgBAKABAaoBC2d3cy13aXotaW1nsAEAwAEB&sclient=img&ei=ENF6X8vJB4yM0AS73YuADw#imgrc=6tpIVvXIcyYlYM)

The main steps involved in image classification techniques are determining a suitable classification system, feature extraction, selecting good training samples, image pre-processing and selection of appropriate classification method, post-classification processing, and finally assessing the overall accuracy. 

**[Detection](https://github.com/Laudarisd/Interview_exam_preparation/tree/main/src/object_detection)**
========================================================================================================

The problem definition of object detection is to determine where objects are located in a given image such as object localisation and which category each object belongs to, i.e. object classification. In simple words, object detection is a type of image classification technique, and besides classifying, this technique also identifies the location of the object instances from a large number of predefined categories in natural images. 

This technique has the capability to search for a specific class of objects, such as cars, people, animals, birds, etc. and has successfully been used in the next-generation image as well as video processing systems. The recent advancements in this technique have only become possible with the advent of deep learning methodologies.


<table border="0">
   <tr>
      <td>
      <img src="./src/img/ob1.jpg" width="100%" />
      </td>
      <td>
      <img src="./src/img/ob2.png" width="200%" />
      </td>
      <td>
      <img src="./src/img/ob3.jpg" width="100%" />
      </td>
      <td>
      <img src="./src/img/ob4.jpg" width="200%" />
      </td>
   </tr>
   </table>

[image-source](https://www.google.com/search?q=object+detection&tbm=isch&source=iu&ictx=1&fir=CeGn9NCnSTk2iM%252CNZgI-_CyMhb-xM%252C_&vet=1&usg=AI4_-kRweDoaQc0az867zaxbCBP27URosg&sa=X&ved=2ahUKEwi52sfA_JzsAhVEMd4KHUdXA90Q_h16BAgLEAU)

Object detection techniques can be used in real-world projects such as face detection, pedestrian detection, vehicle detection, traffic sign detection, video surveillance, among others.  

**How Object Detection Works**
The pipeline of traditional object detection models can be mainly divided into three stages, that are informative region selection, feature extraction and classification. There are several popular deep learning-based models for object detection, which have been used by organisations and academia to achieve efficiency as well as accurate results in detecting objects from images. The popular models include MobileNet, You Only Live Once (YOLO), Mark-RCNN, RetinaNet, among others.

**Disadvantages**
Over the past few years, great success has been achieved in a controlled environment for object detection problem. However, the problem remains unsolved in uncontrolled places, in particular, when objects are placed in arbitrary poses in a cluttered and occluded environment.



* [Common model architectures used for object detection](https://github.com/Laudarisd/Interview_exam_preparation/tree/main/src/object_detection)
* [How do Convolutional Neural Networks work](https://github.com/Laudarisd/Interview_exam_preparation/tree/main/src/object_detection)
* [Features](https://github.com/Laudarisd/Interview_exam_preparation/tree/main/src/object_detection)
* [Convolution](https://github.com/Laudarisd/Interview_exam_preparation/tree/main/src/object_detection)
* [Pooling](https://github.com/Laudarisd/Interview_exam_preparation/tree/main/src/object_detection)
* [Rectified Linear Units(RELU)](https://github.com/Laudarisd/Interview_exam_preparation/tree/main/src/object_detection)
* [Deep learning](https://github.com/Laudarisd/Interview_exam_preparation/tree/main/src/object_detection)
* [Fully connected layers](https://github.com/Laudarisd/Interview_exam_preparation/tree/main/src/object_detection)
* [Backpropagation](https://github.com/Laudarisd/Interview_exam_preparation/tree/main/src/object_detection)
* [Hyperparameters](https://github.com/Laudarisd/Interview_exam_preparation/tree/main/src/object_detection)
* [Deep learning tools](https://github.com/Laudarisd/Interview_exam_preparation/tree/main/src/object_detection)
* [Overview on Frameworks](https://github.com/Laudarisd/Interview_exam_preparation/tree/main/src/object_detection)


