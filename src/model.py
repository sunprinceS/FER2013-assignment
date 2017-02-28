#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Input,Dense,Dropout,Flatten,Activation
from keras.layers import Convolution2D,MaxPooling2D
from keras.optimizers import SGD
nb_class = 7

def build_model():
    # VGG-like convnet:
    model = Sequential()
    model.add(Convolution2D(16,3,3,border_mode='valid',input_shape=(48,48,1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    return model
