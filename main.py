from utils import *

pixels = load_pickle('../fer2013/train_pixels.pkl')
labels = load_pickle('../fer2013/train_labels.pkl')
print (len(pixels))
print (len(labels))
pixels = load_pickle('../fer2013/publicTest_pixels.pkl')
labels = load_pickle('../fer2013/publicTest_labels.pkl')
print (len(pixels))
print (len(labels))
pixels = load_pickle('../fer2013/privateTest_pixels.pkl')
labels = load_pickle('../fer2013/privateTest_labels.pkl')
print (len(pixels))
print (len(labels))
