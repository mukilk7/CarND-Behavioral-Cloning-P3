#!/usr/bin/env python

__author__ = "Mukil Kesavan"

import numpy as np
import math
import pdb
import csv
import cv2
import sys
#
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#
import matplotlib.image as mpimg
#
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

## Global Config Data
DATADIR = "./data/data/"
DRIVING_LOG_FILE = DATADIR + "/driving_log.csv"
IMGDIR = DATADIR + "/IMG/"
LEFT_STEERING_CORRECTION = 0.27
RIGHT_STEERING_CORRECTION = -0.27
#

def buildModel(feature_extract = False, ipshape = (160, 320, 3)):
    """ Builds a deep neural network model with 2 pre-processing
    layers (cropping and normalization), 5 convolutional layers
    non-linearized via the RELU function and 3 fully connected
    layers, again non-linearized via the RELU function. Dropout
    layers of increasing dropout probability as we go deeper in
    the network, have been added after the convolutional layers
    to prevent overfitting.

    The model architecture was originally proposed by Bojarski et.al,
    End to Eng Learning for Self-Driving Cars, of NVIDIA Inc.
    Reference: https://arxiv.org/abs/1604.07316. I've adapted it
    for this project.
    """
    model = Sequential()
    ### Data Pre-processing
    #Cropping input to remove car dash and sky etc.
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=ipshape))
    #Normalization and mean shifting 
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    #Convolutional Layers
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Dropout(0.25))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Dropout(0.3))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Dropout(0.35))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Dropout(0.4))

    model.add(Flatten())
    
    #Fully Connected Layers
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dense(50))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(Activation("relu"))

    #Don't return the output layer if this model
    #is to be used for transfer learning.
    if not feature_extract:
        model.add(Dense(1))

    return model


def getInputSamples(addFlippedImages = False):
    """ Reads the contents of the
    driving log file and returns a
    list of tuples containing the center,
    left and right camera images, and
    the corresponding steering angles. The
    function can optionally flip the input
    images and steering angles and add them
    to the resultant list, to account for
    the left turn bias in track 1.
    """
    samples = []
    with open(DRIVING_LOG_FILE, "r") as dlfile:
        dlreader = csv.reader(dlfile)
        skipHeader = True
        for line in dlreader:
            if skipHeader:
                skipHeader = False
                continue
            #Using images from all 3 cameras
            cfile = line[0].split("\\")[-1]
            lfile = line[1].split("\\")[-1]
            rfile = line[2].split("\\")[-1]
            csteering = float(line[3])
            lsteering = csteering + LEFT_STEERING_CORRECTION
            rsteering = csteering + RIGHT_STEERING_CORRECTION
            #image file, steering input, is image flipped
            samples.append((cfile, csteering, False))
            samples.append((lfile, lsteering, False))
            samples.append((rfile, rsteering, False))
            #append a flipped image/steering input too to account for left turn bias
            #The actual flipping happens in dataGenerator
            if addFlipppedImages:
                samples.append((cfile, csteering * -1.0, True))
                samples.append((lfile, lsteering * -1.0, True))
                samples.append((rfile, rsteering * -1.0, True))
    
    return samples


def dataGenerator(samples, batchsz = 32):
    """ Uses a python generator to return a batch
    tuple of input images and steering angles. It
    also performs the actual flipping of images if
    indicated in the input list.
    """
    while True:
        shuffle(samples)
        for i in range(0, len(samples), batchsz):
            X = []
            y = []
            batchdata = samples[i: i + batchsz]
            for b in batchdata:
                ifile = IMGDIR + b[0]
                img = mpimg.imread(ifile)
                steering = float(b[1])
                #If this was an image intended to be
                #flipped, then do flip. The steering
                #input is already flipped in getInputSamples.
                if b[2] == True:
                    X.append(np.fliplr(img))
                else:
                    X.append(img)
                y.append(steering)
            X = np.array(X)
            y = np.array(y)
            yield shuffle(X, y)


def trainModel(model, samples, valsplit = 0.1, modeloutfile = "./model/model.h5"):
    """ Trains the given model using the given input samples using an AdamOptimizer
    minimizing the mean squared error between input and predicted steering angles.
    The resultant model is also saved to the local filesystem at the end. Finally,
    the function returns a history object with the training metrics (e.g. mse during
    each epoch) that can be used by the caller for plotting.
    """
    train_samples, val_samples = train_test_split(samples, test_size = valsplit)
    trainDataGen = dataGenerator(train_samples, batchsz = 64)
    valDataGen = dataGenerator(val_samples, batchsz = 64)
    model.compile(loss = "mse", optimizer="adam", metrics=["mean_squared_error"])
    history = model.fit_generator(trainDataGen,\
                                  samples_per_epoch = len(train_samples),\
                                  validation_data = valDataGen,\
                                  nb_val_samples = len(val_samples),\
                                  nb_epoch = 25)
    model.save(modeloutfile)
    print("Model saved to \"", modeloutfile, "\"")
    return history

        
def pipeline():
    """ Runs the overall learning pipeline.
    """
    samples = getInputSamples()
    model = buildModel()
    history = trainModel(model, samples)


####

if __name__ == "__main__":
    pipeline()
    
