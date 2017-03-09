*Behavioral Cloning Project*

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./deliverables/writeup_images/p3-nn-model.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of 2 input pre-processing layers (cropping and normalization - model.py lines 48-52), 5 convolutional layers non-linearized via the RELU function (model.py lines 54-64) and 3 fully connected layers, again non-linearized via the RELU function (model.py lines 68-74). For the convolutional layers, filter sizes range from 5x5 to 3x3 with valid padding and output sizes range from 24 to 64. Dropout layers of increasing dropout probability as we go deeper in the network, have been added after each convolutional layers to deal with input noise and prevent overfitting. The model I use is an adapted version of the architecture originally proposed by Bojarski et.al, End to End Learning for Self-Driving Cars, of NVIDIA Inc. Reference: https://arxiv.org/abs/1604.07316.  

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 54-64). The dropout probability was increased linearly as we get deeper into the network as the number of parameters increase in a similar fashion.

I split the input data into 90%-10% test-validation split (respectively) and the model was validated against the validation split to ensure that it doesn't overfit (model.py lines 153-171). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. I also empirically varied the number of training epochs over multiple runs of model.py to figure out where the validation set loss metric starts plateauing substantially. I then built my final model with the number of training epochs set to just before the plateauing starts in order to prevent the network from overfitting to the input.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 163).

####4. Appropriate training data

Training data was generated to keep the vehicle driving on the road. I recorded a full-lap of mostly center lane driving in both clockwise and counter-clockwise directions around the circuit. This was done to workaround the inherent left-turn or right-turn bias in the track in a single direction and generate a balanced dataset. Further, I also used the left and right camera images from each input frame, along with appropriately computed steering angle, in order to incorporate examples of recovery driving.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I initially started with the good old LeNet5 architecture for handwritten digits classification from the previous projects/lessons without any special input pre-processing layers or overfitting controls. I also used the dataset provided by Udacity (center images and steering angles only) to train the network using the adam optimizer for 10-50 epochs with an input batch size of 32. I noticed that the loss metric (mean squared error) stopped improving very early on in the training phase for the validation set while the training set loss kept improving. This meant that the model was overfitting fast. I realized that I needed to add dropout to the model and did so. I also realized that there was probably some input pre-processing that could be done to reduce the input noise. So I added a cropping and a normalization layer early on to the model.

These changes helped somewhat but the car still wasn't able to correct itself once it started straying. It was at this point that I started suspecting that perhaps a more complex model architecture might be required to represent the concepts involved in predicting the steering angle (purely intuition, I should say). So I implemented the model from the paper Bojarski et.al., End to Eng Learning for Self-Driving Cars, of NVIDIA Inc., with my pre-processing and dropout adaptations. This model performed substantially better and the car was able to drive a substantially larger portion of the circuit without veering off-road. I realized at this point that if I augmented the training data to account for the left turn bias and included recovery driving, I might get the car to complete a full lap properly. I did and it worked as shown in the final video. The data augmentation and training process is explained next.

####2. Final Model Architecture

The final model architecture (model.py lines 33-81) consisted of a convolution neural network as shown below:

![alt text][image1]

The layer parameters and activation functions are shown in the diagram. In addition, for the convolutional layers, I used "valid" padding with a stride of (2,2) for the layers with 5x5 filters and a stride of (1,1) for the layers with 3x3 filters, similar to the NVIDIA paper referenced earlier.

####3. Creation of the Training Set & Training Process

I did the following things to create my entire input dataset:

* I drove around the track in counter-clockwise direction around the circuit - this resulted in a dataset that had a left-turn bias.
* I then turned my car around and drove around the circuit in clockwise direction - this resulted in a dataset that had a right-turn bias.
* I combined the images and steering angles from both the drives above so that I get a balanced dataset in terms of turn-bias.
* In order to include examples of recovery driving, I basically used the left and right images from the combined dataset and computed the appropriate steering angle from the steering angle for the center image (model.py lines 103 - 113) as follows:

right_img_steering_angle = center_img_steering_angle - CORRECTION_FACTOR
left_img_steering_angle = center_img_steering_angle + CORRECTION_FACTOR

Basically the idea here is that in the image produced by the right camera, the location that the car needs to get to as per the center camera image prediction, appears further left. Or equivalently the car is veering to the right relative to its actual position. Therefore, we need to subtract a constant factor from the steering angle to get the car to turn further left to get to the correct position. A similar explanation works for the left camera image with the sign of the correction factor reversed. I've illustrated the process below:




I experimented with values for the correction_factor between 0.15 and 0.35 (binary searched through the space) and found out that a value of 0.27 worked well in terms of recovering the car without being too agressive and bumping into things.

