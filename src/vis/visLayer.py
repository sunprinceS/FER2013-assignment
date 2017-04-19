#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from utils import *
from marcos import *

def main():
    emotion_classifier = load_model(model_path)
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

    input_img = emotion_classifier.input
    collect_layers = list()
    collect_layers.append(K.function([input_img,K.learning_phase()],[layer_dict['visLayer'].output]))

    dev_feat,dev_label,dev_num = read_dataset('valid')
    choose_id = 17
    photo = dev_feat[choose]
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo.reshape(1,48,48,1),0]) #get the output of that layer
        fig = plt.figure(figsize=(14,8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16,16,i+1)
            ax.imshow(im[0][0,:,:,i],cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt,choose_id))
        img_path = os.path.join(vis_dir,store_path)
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))
