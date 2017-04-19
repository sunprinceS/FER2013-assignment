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

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    # return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
    mm = K.max(x)
    return x/mm

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_dir = os.path.join(base_dir,'exp')
img_dir = os.path.join(base_dir,'image')
cmap_dir = os.path.join(img_dir,'cmap')
cmap_dir = os.path.join(img_dir,'parital_see')

def main():
    parser = argparse.ArgumentParser(prog='attention_maps.py',
            description='ML-Assignment3 visualize attention heat map.')
    parser.add_argument('--model',type=str,default='simple',choices=['simple','NTUEE'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=80)
    parser.add_argument('--idx',type=int,metavar='<suffix>',required=True)
    args = parser.parse_args()
    store_path = "{}_epoch{}_{}".format(args.model,args.epoch,args.idx)
    print(colored("Loading model from {}".format(store_path),'yellow',attrs=['bold']))
    model_path = os.path.join(exp_dir,store_path,'model.h5')

    emotion_classifier = load_model(model_path)
    input_img = emotion_classifier.input
    te_feats,te_labels,_ = read_dataset('privateTest')
    img_ids = [17]

    layer_idx = [idx for idx, layer in enumerate(emotion_classifier.layers) if layer.name == 'activation_1'][0]

    for idx in img_ids:
        pred =emotion_classifier.predict_classes(te_feats[idx].reshape((1,48,48,1)))
        target = K.mean(emotion_classifier.output[:,pred])
        grads = K.gradients(target,input_img)[0]

        fn = K.function([input_img,K.learning_phase()],[grads])
        input_img_data = te_feats[idx].reshape(1,48,48,1)
        
        heatmap = fn([input_img_data,0])[0].reshape(48,48)
        heatmap = deprocess_image(heatmap)
        heatmap /= np.max(heatmap)

        see = te_feats[idx].reshape(48,48)
        see[np.where(heatmap <= 0.5)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap,cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(img_dir,'cmap','privateTest','{}.png'.format(idx)),dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(img_dir,'partial_see','privateTest','{}.png'.format(idx)),dpi=100)

if __name__ == "__main__":
    main()
