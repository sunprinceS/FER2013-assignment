#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
img_dir = os.path.join(base_dir,'image')
cmap_dir = os.path.join(img_dir,'cmap')
combine_dir = os.path.join(img_dir,'combine')

def main():
    img_idx = [5,8,13,14]
    for idx in img_idx:
        orig = cv2.imread(os.path.join(img_dir,'privateTest','{}.png'.format(idx)))
        cmap = cv2.imread(os.path.join(cmap_dir,'privateTest','{}.png'.format(idx)))
        # print(type(img1))
        
        result = cv2.addWeighted(cmap,1,orig,1,0.5)
        cv2.imwrite(os.path.join(combine_dir,'privateTest','{}.png'.format(idx)),result)



if __name__ == "__main__":
    main()
