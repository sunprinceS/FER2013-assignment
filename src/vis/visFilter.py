#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def grad_ascent(num_step,input_image_data,iter_func):
    """
    You should implement this function
    """
    return input_img_data

def main():
    emotion_classifier = load_model(model_path)
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    input_img = emotion_classifier.input

    name_ls = ["the layers' names you want to get their output"]
    collect_layers = list()
    collect_layers.append(layer_dict[name_ls[0]].output)

    for cnt, c in enumerate(collect_layers):
        filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
        for filter_idx in range(nb_filter):
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            target = K.mean(c[:,:,:,filter_idx])
            grads = normalize(K.gradients(target,input_img)[0])
            iterate = K.function([input_img],[target,grads])
            
            grad_ascent(num_step,input_img_data,iterate)

            # for it in range(NUM_STEPS):
                # target_val, grads_val = iterate([input_img_data])
                # input_img_data += grads_val*LR_RATE
                # if it % RECORD_FREQ == 0:
                tmp = input_img_data.reshape(48,48,1)
                filter_imgs[it//RECORD_FREQ].append((deprocess_image(tmp).reshape(48,48),target_val))

        for it in range(NUM_STEPS//RECORD_FREQ):
            fig = plt.figure(figsize=(14,8))
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/16,16,i+1)
                ax.imshow(filter_imgs[it][i][0],cmap='BuGn')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.xlabel('{:.3f}'.format(filter_imgs[it][i][1]))
                plt.tight_layout()
            fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[0],it*RECORD_FREQ))
            img_path = os.path.join(filter_dir,'{}-{}'.format(store_path,name_ls[0]))
            if not os.path.isdir(img_path):
                os.mkdir(img_path)
            fig.savefig(os.path.join(img_path,'e{}'.format(it*RECORD_FREQ)))
