#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
from utils import *
import numpy as np
import matplotlib.pyplot as plt

from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_dir = os.path.join(base_dir,'exp')
img_dir = os.path.join(base_dir,'image')

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
    img_ids = [17]
    for idx in img_ids:
        # seed_img = vis_utils.load_img(os.path.join(img_dir,'{}.png'.format(img_idx)))
        pred =emotion_classifier.predict_classes(te_feats[idx].reshape((1,48,48,1)))
        heatmap = visualize_saliency(emotion_classifier,layer_idx,[pred],te_feats[idx]).reshape(48,48)
        # cam = visualize_cam(emotion_classifier, layer_idx, [pred], te_feats[idx]).reshape(48,48)
        see = te_feats[idx].reshape(48,48)
        # see_cam = te_feats[idx].reshape(48,48)
        see[np.where(heatmap <= 0.5)] = np.mean(see)

        # see_cam[np.where(cam<= 0.3)] = np.mean(see)

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

        # plt.figure()
        # plt.imshow(cam,cmap=plt.cm.jet)
        # plt.colorbar()
        # plt.tight_layout()
        # fig = plt.gcf()
        # plt.draw()
        # fig.savefig(os.path.join(img_dir,'cam','privateTest','{}.png'.format(idx)),dpi=100)

        # plt.figure()
        # plt.imshow(see_cam,cmap='gray')
        # plt.colorbar()
        # plt.tight_layout()
        # fig = plt.gcf()
        # plt.draw()
        # fig.savefig(os.path.join(img_dir,'partial_cam','privateTest','{}.png'.format(idx)),dpi=100)


if __name__ == '__main__':
    main()
    # generate_saliceny_map()
