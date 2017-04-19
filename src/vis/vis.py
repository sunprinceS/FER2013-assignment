#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from termcolor import colored,cprint
import numpy as np
from utils import *

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_dir = os.path.join(base_dir,'exp')
o_act_dir = os.path.join(base_dir,'image','o_act')
vis_dir = os.path.join(base_dir,'image','vis_layer')
filter_dir = os.path.join(base_dir,'image','vis_filter')

nb_class = 7
LR_RATE = 2 * 1e-2
NUM_STEPS = 200
RECORD_FREQ = 10

def main():
    parser = argparse.ArgumentParser(prog='visFilter.py',
            description='Visualize CNN filter.')
    parser.add_argument('--model',type=str,default='simple',choices=['simple','NTUEE'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=20)
    parser.add_argument('--mode',type=int,metavar='<visMode>',default=2,choices=[1,2,3])
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    parser.add_argument('--idx',type=int,metavar='<suffix>',required=True)
    args = parser.parse_args()
    store_path = "{}_epoch{}_{}".format(args.model,args.epoch,args.idx)
    print(colored("Loading model from {}".format(store_path),'yellow',attrs=['bold']))
    model_path = os.path.join(exp_dir,store_path,'model.h5')
    emotion_classifier = load_model(model_path)


    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


    input_img = emotion_classifier.input

    # Visualize most activated image for each class

    if args.mode == 1:
        total_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]

        for o_class in range(nb_class):
            target_loss =  -K.mean(emotion_classifier.output[:,o_class])
            # target_loss = normalize(target_loss)
            a = K.square(input_img[:, :48- 1, :48 - 1, :] - input_img[:, 1:, :48 - 1, :])
            b = K.square(input_img[:, :48- 1, :48 - 1, :] - input_img[:, :48 - 1, 1:, :])
            tv_loss = K.sum(K.pow(a + b, 1.25))
            loss = target_loss * 5 + tv_loss


            # compute the gradient of the input picture
            grads = K.gradients(loss, input_img)[0]
            # print(grads.shape) #(?,48,48,1)

            # normalization trick: we normalize the gradient
            grads = normalize(grads)

            # this function returns the loss and grads given the input picture
            iterate = K.function([input_img, K.learning_phase()], [loss, grads])

            input_img_data = np.random.random((1, 48, 48, 1))

            for it in range(NUM_STEPS):
                loss , grads_value = iterate([input_img_data,0]) # 0: not training
                input_img_data -= grads_value *  LR_RATE
                if it % RECORD_FREQ == 0:
                    print(loss)
                    tmp = input_img_data.reshape(48,48,1)
                    total_imgs[it//RECORD_FREQ].append(deprocess_image(tmp).reshape(48,48))
            print('*'*30)

            # input_img_data = input_img_data.reshape(48,48,1)
            # img = deprocess_image(input_img_data)
            # ex_imgs.append(img.reshape(48,48))

        for it in range(NUM_STEPS//RECORD_FREQ):
            fig = plt.figure(figsize=(8,2))
            for o_class in range(7):
                ax = fig.add_subplot(1,7,o_class+1)
                ax.imshow(total_imgs[it][o_class],cmap='gray')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.tight_layout()
                plt.xlabel(classes[o_class])
            fig.suptitle('The most activated image of each class (# Ascent Epoch: {})'.format(it*RECORD_FREQ))
            img_path = os.path.join(o_act_dir,'{}_tv'.format(store_path))
            if not os.path.isdir(img_path):
                os.mkdir(img_path)
            fig.savefig(os.path.join(img_path,'tv_act_class_e{}'.format(it*RECORD_FREQ)))


    # visualize the area CNN see
    elif args.mode == 2:
        collect_layers = list()
        # collect_layers.append(K.function([input_img,K.learning_phase()],[layer_dict['convolution2d_2'].output]))
        collect_layers.append(K.function([input_img,K.learning_phase()],[layer_dict['zeropadding2d_1'].output]))
        collect_layers.append(K.function([input_img,K.learning_phase()],[layer_dict['zeropadding2d_3'].output]))
        collect_layers.append(K.function([input_img,K.learning_phase()],[layer_dict['zeropadding2d_4'].output]))

        dev_feat,dev_label,_ = read_dataset('privateTest')
        choose_id =17
        photo = dev_feat[choose_id]
        for cnt, fn in enumerate(collect_layers):
            im = fn([photo.reshape(1,48,48,1),0])
            fig = plt.figure(figsize=(14,8))
            nb_filter = im[0].shape[3]
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/16,16,i+1)
                ax.imshow(im[0][0,:,:,i],cmap='Purples')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.tight_layout()
            fig.suptitle('Output of layer{} (Given image{})'.format(cnt,choose_id))
            img_path = os.path.join(vis_dir,store_path)
            if not os.path.isdir(img_path):
                os.mkdir(img_path)
            fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))

    else:
        # name_ls = ['zeropadding2d_1','zeropadding2d_3','zeropadding2d_4']
        name_ls = ['zeropadding2d_4']
        collect_layers = list()
        collect_layers.append(layer_dict[name_ls[0]].output)
        # collect_layers.append(layer_dict[name_ls[1]].output)
        # collect_layers.append(layer_dict[name_ls[2]].output)

        for cnt, c in enumerate(collect_layers):
            filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
            print(cnt)
            for filter_idx in range(128):
                input_img_data = np.random.random((1, 48, 48, 1))
                loss = K.mean(c[:,:,:,filter_idx])
                grads = normalize(K.gradients(loss,input_img)[0])
                iterate = K.function([input_img],[loss,grads])
                
                for it in range(NUM_STEPS):
                    loss_val, grads_val = iterate([input_img_data])
                    input_img_data += grads_val * LR_RATE
                    if it % RECORD_FREQ == 0:
                        tmp = input_img_data.reshape(48,48,1)
                        filter_imgs[it//RECORD_FREQ].append((deprocess_image(tmp).reshape(48,48),loss_val))
            for it in range(NUM_STEPS//RECORD_FREQ):
                fig = plt.figure(figsize=(14,8))
                for i in range(128):
                    ax = fig.add_subplot(128/16,16,i+1)
                    ax.imshow(filter_imgs[it][i][0],cmap='Purples')
                    plt.xticks(np.array([]))
                    plt.yticks(np.array([]))
                    plt.xlabel('{:.3f}'.format(filter_imgs[it][i][1]))
                    plt.tight_layout()
                fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[0],it*RECORD_FREQ))
                img_path = os.path.join(filter_dir,'{}-{}'.format(store_path,name_ls[0]))
                if not os.path.isdir(img_path):
                    os.mkdir(img_path)
                fig.savefig(os.path.join(img_path,'e{}'.format(it*RECORD_FREQ)))

if __name__ == "__main__":
    main()
