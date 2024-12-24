Difference Between Image Classification & Object Detection
============================================================

**Image Classification**

In simple words, image classification is a technique that is used to classify or predict the class of a specific object in an image. The main goal of this technique is to accurately identify the features in an image.

How Image Classification Works
In general, the image classification techniques can be categorised as parametric and non-parametric or supervised and unsupervised as well as hard and soft classifiers. For supervised classification, this technique delivers results based on the decision boundary created, which mostly rely on the input and output provided while training the model. But, in the case of unsupervised classification, the technique provides the result based on the analysis of the input dataset own its own; features are not directly fed to the models.

The main steps involved in image classification techniques are determining a suitable classification system, feature extraction, selecting good training samples, image pre-processing and selection of appropriate classification method, post-classification processing, and finally assessing the overall accuracy. 

**Detection**
The problem definition of object detection is to determine where objects are located in a given image such as object localisation and which category each object belongs to, i.e. object classification. In simple words, object detection is a type of image classification technique, and besides classifying, this technique also identifies the location of the object instances from a large number of predefined categories in natural images. 

This technique has the capability to search for a specific class of objects, such as cars, people, animals, birds, etc. and has successfully been used in the next-generation image as well as video processing systems. The recent advancements in this technique have only become possible with the advent of deep learning methodologies.

Object detection techniques can be used in real-world projects such as face detection, pedestrian detection, vehicle detection, traffic sign detection, video surveillance, among others.  

How Object Detection Works
The pipeline of traditional object detection models can be mainly divided into three stages, that are informative region selection, feature extraction and classification. There are several popular deep learning-based models for object detection, which have been used by organisations and academia to achieve efficiency as well as accurate results in detecting objects from images. The popular models include MobileNet, You Only Live Once (YOLO), Mark-RCNN, RetinaNet, among others.

Disadvantages 
Over the past few years, great success has been achieved in a controlled environment for object detection problem. However, the problem remains unsolved in uncontrolled places, in particular, when objects are placed in arbitrary poses in a cluttered and occluded environment.


Common model architectures used for object detection
=======================================================

* R-CNN
* Fast R-CNN
* Faster R-CNN
* Mask R-CNN
* SSD (Single Shot MultiBox Defender)
* YOLO (You Only Look Once)
* Objects as Points
* Data Augmentation Strategies for Object Detection 

[References](https://heartbeat.fritz.ai/a-2019-guide-to-object-detection-9509987954c3)

How do Convolutional Neural Networks work
===========================================

* Features

CNNs compare images piece by piece. The piece that it looks for are called features. Each features is like mini-image which is a small two dimentional array of values.
<table border="0">
   <tr>
      <td>
      <img src="./Computer-Vision/Project-1.Study_%26_Research/ML/img/features.png" width="100%" />
      </td>
   </tr>
   </table>


* Convolution 
When presented with a new image, the CNN doesn’t know exactly where these features will match so it tries them everywhere, in every possible position. In calculating the match to a feature across the whole image, we make it a filter. The math we use to do this is called convolution, from which Convolutional Neural Networks take their name.

<table border="0">
   <tr>
      <td>
      <img src="./img/convolution.png" width="100%" />
      </td>
   </tr>
   </table>

The math behind convolution is nothing that would make a sixth-grader uncomfortable. To calculate the match of a feature to a patch of the image, simply multiply each pixel in the feature by the value of the corresponding pixel in the image. Then add up the answers and divide by the total number of pixels in the feature. If both pixels are white (a value of 1) then 1 * 1 = 1. If both are black, then (-1) * (-1) = 1. Either way, every matching pixel results in a 1. Similarly, any mismatch is a -1. If all the pixels in a feature match, then adding them up and dividing by the total number of pixels gives a 1. Similarly, if none of the pixels in a feature match the image patch, then the answer is a -1.

* Pooling

Another power tool that CNNs use is called pooling. Pooling is a way to take large images and shrink them down while preserving the most important information in them. The math behind pooling is second-grade level at most. It consists of stepping a small window across an image and taking the maximum value from the window at each step. In practice, a window 2 or 3 pixels on a side and steps of 2 pixels work well.

After pooling, an image has about a quarter as many pixels as it started with. Because it keeps the maximum value from each window, it preserves the best fits of each feature within the window. This means that it doesn’t care so much exactly where the feature fit as long as it fit somewhere within the window. The result of this is that CNNs can find whether a feature is in an image without worrying about where it is. This helps solve the problem of computers being hyper-literal.

<table border="0">
   <tr>
      <td>
      <img src="./img/pooling.png" width="100%" />
      </td>
   </tr>
   </table>


* Rectified Linear Units

A small but important player in this process is the Rectified Linear Unit or ReLU. It’s math is also very simple—wherever a negative number occurs, swap it out for a 0. This helps the CNN stay mathematically healthy by keeping learned values from getting stuck near 0 or blowing up toward infinity. It’s the axle grease of CNNs—not particularly glamorous, but without it they don’t get very far.

<table border="0">
   <tr>
      <td>
      <img src="./img/rlu1.png" width="100%" />
      </td>
      <td>
      <img src="./img/rlu2.png" width="100%" />
      </td>
   </tr>
   </table>

The output of a ReLU layer is the same size as whatever is put into it, just with all the negative values removed.

* Deep learning

You’ve probably noticed that the input to each layer (two-dimensional arrays) looks a lot like the output (two-dimensional arrays). Because of this, we can stack them like Lego bricks. Raw images get filtered, rectified and pooled to create a set of shrunken, feature-filtered images. These can be filtered and shrunken again and again. Each time, the features become larger and more complex, and the images become more compact. This lets lower layers represent simple aspects of the image, such as edges and bright spots. Higher layers can represent increasingly sophisticated aspects of the image, such as shapes and patterns. These tend to be readily recognizable. For instance, in a CNN trained on human faces, the highest layers represent patterns that are clearly face-like.

<table border="0">
   <tr>
      <td>
      <img src="./img/deep1.png" width="100%" />
      </td>
      <td>
      <img src="./img/deep2.png" width="100%" />
      </td>
   </tr>
   </table>

* Fully connected layers

CNNs have one more arrow in their quiver. Fully connected layers take the high-level filtered images and translate them into votes. In our case, we only have to decide between two categories, X and O. Fully connected layers are the primary building block of traditional neural networks. Instead of treating inputs as a two-dimensional array, they are treated as a single list and all treated identically. Every value gets its own vote on whether the current image is an X or and O. However, the process isn’t entirely democratic. Some values are much better than others at knowing when the image is an X, and some are particularly good at knowing when the image is an O. These get larger votes than the others. These votes are expressed as weights, or connection strengths, between each value and each category.

When a new image is presented to the CNN, it percolates through the lower layers until it reaches the fully connected layer at the end. Then an election is held. The answer with the most votes wins and is declared the category of the input.

<table border="0">
   <tr>
      <td>
      <img src="./img/fc1.png" width="100%" />
      </td>
      <td>
      <img src="./img/fc2.png" width="100%" />
      </td>
   </tr>
   </table>


* Backpropagation

Our story is filling in nicely, but it still has a huge hole—Where do features come from? and How do we find the weights in our fully connected layers? If these all had to be chosen by hand, CNNs would be a good deal less popular than they are. Luckily, a bit of machine learning magic called backpropagation does this work for us.

<table border="0">
   <tr>
      <td>
      <img src="./img/bp1.png" width="100%" />
      </td>
   </tr>
   </table>

for more information [click](https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c#:~:text=Chain%20Rule%20in%20a%20Convolutional%20Layer&text=For%20the%20forward%20pass%2C%20we,as%20%E2%88%82L%2F%E2%88%82z.)


* Hyperparameters

Unfortunately, not every aspect of CNNs can be learned in so straightforward a manner. There is still a long list of decisions that a CNN designer must make.

For each convolution layer, How many features? How many pixels in each feature?
For each pooling layer, What window size? What stride?
For each extra fully connected layer, How many hidden neurons?
In addition to these there are also higher level architectural decisions to make: How many of each layer to include? In what order? Some deep neural networks can have over a thousand layers, which opens up a lot of possibilities.



Deep learning tools
====================

* Caffe
* CNTK
* Deeplearning4j
* TensorFlow
* Theano
* Torch
* Many others

for more [click](https://e2eml.school/how_convolutional_neural_networks_work.html#:~:text=Each%20image%20the%20CNN%20processes%20results%20in%20a%20vote.&text=After%20doing%20this%20for%20every,the%20set%20of%20labeled%20images.)



|Label     |	Probability|
| :-------------: | :----------: |
|rabbit  |	0.07|
|hamster |	0.02|
|dog     |	0.91 | 
