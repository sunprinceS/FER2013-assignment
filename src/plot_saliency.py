#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-7)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    # x *= 255
    # x = x.transpose((1, 2, 0))
    # x = np.clip(x, 0, 255).astype('uint8')
    return x

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
model_dir = os.path.join(base_dir, 'model')

def main():
    parser = argparse.ArgumentParser(prog='plot_saliency.py',
            description='ML-Assignment3 visualize attention heat map.')
    parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=80)
    args = parser.parse_args()
    model_name = "model-%s.h5" %str(args.epoch)
    model_path = os.path.join(model_dir, model_name)
    emotion_classifier = load_model(model_path)
    print(colored("Loaded model from {}".format(model_name), 'yellow', attrs=['bold']))

    private_pixels = load_pickle('../fer2013/privateTest_pixels.pkl')
    private_pixels = [ np.fromstring(private_pixels[i], dtype=float, sep=' ').reshape((1, 48, 48, 1)) 
                       for i in range(len(private_pixels)) ]
    input_img = emotion_classifier.input
    img_ids = [17]

    for idx in img_ids:
        val_proba = emotion_classifier.predict(private_pixels[idx])
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        input_img_data = private_pixels[idx]
        heatmap = fn([input_img_data, 0])[0].reshape(48, 48)
        heatmap = np.abs(heatmap)
        heatmap = deprocess_image(heatmap)
        print heatmap
        heatmap /= np.max(heatmap)

        thres = 0.5
        see = private_pixels[idx].reshape(48, 48)
        # for i in range(48):
            # for j in range(48):
                # print heatmap[i][j]
        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(cmap_dir, 'privateTest', '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(partial_see_dir, 'privateTest', '{}.png'.format(idx)), dpi=100)

if __name__ == "__main__":
    main()
