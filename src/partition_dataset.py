import cPickle as pickle

csvFile = '../fer2013/fer2013.csv'
pixelsFile = ['../fer2013/train_pixels.pkl', '../fer2013/publicTest_pixels.pkl', '../fer2013/privateTest_pixels.pkl']
labelsFile = ['../fer2013/train_labels.pkl' ,'../fer2013/publicTest_labels.pkl' ,'../fer2013/privateTest_labels.pkl' ]

pixels = [[], [], []]
labels = [[], [], []]

with open(csvFile, 'r') as f:
    f.readline()
    for i, line in enumerate(f):
        data = line.split(',')
        label = data[0]
        pixel = data[1]
        usage = data[2][:-1]
        # print usage
        if usage == 'Training':
            pixels[0].append(pixel)
            labels[0].append(label)
        elif usage == 'PublicTest':
            pixels[1].append(pixel)
            labels[1].append(label)
        elif usage == 'PrivateTest':
            pixels[2].append(pixel)
            labels[2].append(label)

for i in range(3):
    with open(pixelsFile[i], 'wb') as f:
        pickle.dump(pixels[i], f, pickle.HIGHEST_PROTOCOL)
        print ('Saved pixelsFile %d' %i)
    with open(labelsFile[i], 'wb') as f:
        pickle.dump(labels[i], f, pickle.HIGHEST_PROTOCOL)
        print ('Saved labelsFile %d' %i)
