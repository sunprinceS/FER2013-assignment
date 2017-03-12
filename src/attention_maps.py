#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
from utils import *
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

from vis.utils import utils as vis_utils
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_dir = os.path.join(base_dir,'exp')
img_dir = os.path.join(base_dir,'image')

def generate_saliceny_map():

    # Build the VGG16 network with ImageNet weights
    # model = VGG16(weights='imagenet', include_top=True)
    # print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'activation_1'
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

    for path in ['../resources/ouzel.jpg', '../resources/ouzel_1.jpg']:
        seed_img = vis_utils.load_img(path, target_size=(224, 224))
        pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))
        heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img)
        cv2.imshow('Saliency - {}'.format(vis_utils.get_imagenet_label(pred_class)), heatmap)
        cv2.waitKey(0)


def generate_cam():
    """Generates a heatmap via grad-CAM method.
    First, the class prediction is determined, then we generate heatmap to visualize that class.
    """
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

    for path in ['https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Tigerwater_edit2.jpg/170px-Tigerwater_edit2.jpg']:
        seed_img = vis_utils.load_img(path, target_size=(224, 224))
        pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))
        heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img)
        cv2.imshow('Attention - {}'.format(vis_utils.get_imagenet_label(pred_class)), heatmap)
        cv2.waitKey(0)

def main():
    parser = argparse.ArgumentParser(prog='attention_maps.py',
            description='FER2013 visualize attention heat map.')
    parser.add_argument('--model',type=str,default='simple',choices=['simple','NTUEE'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=60)
    parser.add_argument('--idx',type=int,metavar='<suffix>',required=True)
    args = parser.parse_args()
    store_path = "{}_epoch{}_{}".format(args.model,args.epoch,args.idx)
    print(colored("Loading model from {}".format(store_path),'yellow',attrs=['bold']))
    model_path = os.path.join(exp_dir,store_path,'model.h5')

    emotion_classifier = load_model(model_path)
    emotion_classifier.summary()

    layer_idx = [idx for idx, layer in enumerate(emotion_classifier.layers) if layer.name == 'activation_1'][0]
    te_feats,te_labels,_ = read_dataset('privateTest')
    img_ids = [5,8,13,14]
    for idx in img_ids:
        # seed_img = vis_utils.load_img(os.path.join(img_dir,'{}.png'.format(img_idx)))
        pred =emotion_classifier.predict_classes(te_feats[idx].reshape((1,48,48,1)))
        heatmap = visualize_saliency(emotion_classifier,layer_idx,[pred],te_feats[idx]).reshape(48,48)
        plt.figure()
        plt.imshow(heatmap,cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(img_dir,'cmap','privateTest','{}.png'.format(idx)),dpi=100)


if __name__ == '__main__':
    main()
    # generate_saliceny_map()
