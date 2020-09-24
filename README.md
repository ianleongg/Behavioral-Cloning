# **Behavioral Cloning** 

##  Deep neural networks and convolutional neural networks to clone driving behavior 

### Train, validate and test a CNN model using Keras
---

**Behavaral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./Images_forReadMe/birds_eye_view.gif "bird eye view"
[image2]: ./Images_forReadMe/center.jpg "center"
[image3]: ./Images_forReadMe/center_flip.jpg "center flip"
[image4]: ./Images_forReadMe/combined.png "combined"
[image5]: ./Images_forReadMe/combined_crop.png "combined crop"
[image6]: ./Images_forReadMe/drivers_view.gif "drivers view"
[image7]: ./Images_forReadMe/model.png "model"
[image8]: ./Images_forReadMe/left_flip.jpg "left flip"
[image9]: ./Images_forReadMe/right.jpg "right"
[image10]: ./Images_forReadMe/right_flip.png "right flip"
[image11]: ./Images_forReadMe/sample_csv.png "csv"
[image12]: ./Images_forReadMe/layer.png "layer"
[image13]: ./Images_forReadMe/left.jpg "left"


---
### README

- A neural network model was trained and then used to drive the car autonomously around the track.

- There are 4 important files in this project:

  * [model.py](./model.py) (script used to create and train the model)
  * [drive.py](./drive.py) (script to drive the car in autonomous mode)
  * [model.h5](./model.h5) (a trained Keras CNN model)
  * [driving_log.csv](./data/driving_log.csv) (data used to train Keras model)

- Below is the result for the CNN model to output a steering angle to an autonomous vehicle at its max speed setting (~30mph):

Birds-eye View:

![alt text][image1] 

Drivers View:

![alt text][image6]

### Repo contained

#### 1. Functional code

* [model.py](./model.py) file contains the code for training and saving the convolution neural network [(saved as model.h5)](./model.h5). The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

* Using the Udacity provided simulator and my [drive.py](./drive.py) file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

### Model Architecture 

#### 1. Solution Design Approach

* My [model](./model.py) was based off the [NVIDIA's "End to End Learning for Self-Driving Cars" paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) and consists of a convolution neural network with three 5x5 convolution layers, two 3x3 convolution layers, and five fully connected layers.

* The model includes RELU layers to introduce nonlinearity, the data is normalized in the model using a Keras lambda layer, and Max Pooling layers to progressively reduce the spatial size of the representation by reducing the amount of parameters and computation in the network.

* The model architecture is seen as the following:

![alt text][image12]

#### 2. Attempts to reduce overfitting in the model

* The model was trained and validated on different data sets to ensure that the model was not overfitting. It was splitted into a 80:20 ratio with the train_test_split function in sklearnn.model_selection. The data sets can be found [here](./data)

  - Number of training samples:  6428
  - Number of validation samples:  1608

* The ideal number of epochs was 10 as evidenced by a mean squared error loss graph to avoid underfitting/ overfitting.

![alt text][image7]

* The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 									|
 
#### 3. Model parameter tuning

* The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

* Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

* For details about how I created the training data, see the next section. 
 
### Training Data Strategy

#### 1. Creation of the Training Set & Training Process

* All the training set data can be found in the [data file](./data). In this file, contains:
  - [images](./data/IMG)
  - [csv file for driving log](./data/driving_log.csv)

* The simulator captures images from three cameras mounted on the car: a center, right and left camera with its steering, throttle commands for each frame recorded. An example below can be seen where it contains the left, center, and right view in order:

![alt text][image4]

* The first 10 recorded data in the csv file can be seen as the following:

![alt text][image11]

* To augment the data set, I also flipped images and angles thinking that this would increase the training set available. For example, here are images that has then been flipped:

Left original:

![alt text][image13]

Left flipped:

![alt text][image8]

Center original:

![alt text][image2]

Center flipped:

![alt text][image3]

Right original:

![alt text][image9]

Right flipped:

![alt text][image10]

* By using not only the center image but with also both left and right images, the training set data was also increased.
  - From the perspective of the left camera, the steering angle would be less than the steering angle from the center camera.
  - From the right camera's perspective, the steering angle would be larger than the angle from the center camera
  - An arbitrary number of 0.25 was used for the above correction: 
    - left_angle = center_angle + correction
    - right_angle = center_angle - correction

* The data for all 3 camera view images were also cropped by 52px(top) and 23px(bottom) to focus just on the road as seen: 

![alt text][image5]

### Conclusion

* Ultimately, it was a great exposure to use train, test, and validate a CNN model with Keras.
* The data can be pre-processed even further which includes manipulating color spaces, etc.
* The final product can be seen below:

[Birds-Eye View](./Images_forReadMe/birds_eye_view.mp4)
[Drivers View](./Images_forReadMe/drivers_view.mp4)

