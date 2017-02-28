#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Input,Dense,Dropout,Flatten,Activation
from keras.layers import Convolution2D,MaxPooling2D
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import ZeroPadding2D,AveragePooling2D
from keras.optimizers import SGD, Adadelta, Adam
from termcolor import colored, cprint
nb_class = 7

def build_model(mode):
    
    print(colored("Use model {}".format(mode),'yellow',attrs=['bold']))
    model = Sequential()
    if mode == 'simple':
        # VGG-like convnet:
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
        opt = Adam(lr=0.01,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0)
        # opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
    elif mode == 'NTUEE':
        model.add(Convolution2D(64,5,5,border_mode='valid', input_shape=(48,48,1)))
        model.add(PReLU(init='zero',weights=None))
        model.add(ZeroPadding2D(padding=(2,2),dim_ordering='tf'))
        model.add(MaxPooling2D(pool_size=(5,5),strides=(2,2)))

        model.add(ZeroPadding2D(padding=(1,1),dim_ordering='tf'))
        model.add(Convolution2D(64,3,3))
        model.add(PReLU(init='zero',weights=None))
        model.add(AveragePooling2D(pool_size=(3,3),strides=(2,2)))
        model.add(ZeroPadding2D(padding=(1,1),dim_ordering='tf'))
        model.add(Convolution2D(128,3,3))
        model.add(PReLU(init='zero',weights=None))
        model.add(ZeroPadding2D(padding=(1,1),dim_ordering='tf'))
        model.add(Convolution2D(128,3,3))
        model.add(PReLU(init='zero',weights=None))
        model.add(ZeroPadding2D(padding=(1,1),dim_ordering='tf'))
        model.add(AveragePooling2D(pool_size=(3,3),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(PReLU(init='zero',weights=None))
        model.add(Dropout(0.2))
        model.add(Dense(1024))
        model.add(PReLU(init='zero', weights=None))
        model.add(Dropout(0.2))
        model.add(Dense(nb_class))
        model.add(Activation('softmax'))

        opt= Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
    model.summary()
    return model

