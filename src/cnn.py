from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adadelta

def build_model():
	
    input_img = Input(shape=(48, 48, 1))
    # norm_img = BatchNormalization()(input_img)
    # block1 = Convolution2D(64, 5, 5, border_mode='valid')(norm_img)
    '''
    # basic vgg-like
    block1 = Conv2D(32, 3, 3, activation='relu')(input_img)
    block1 = Conv2D(32, 3, 3, activation='relu')(block1)
    block1 = MaxPooling2D((2, 2))(block1)
    block1 = Dropout(0.25)(block1)
 
    block2 = Conv2D(64, 3, 3, activation='relu')(block1)
    block2 = Conv2D(64, 3, 3, activation='relu')(block2)
    block2 = MaxPooling2D((2, 2))(block2)
    block2 = Dropout(0.25)(block2)
 
    block3 = Flatten()(block2)
    block3 = Dense(256, activation='relu')(block3)
    block3 = Dropout(0.5)(block3)
    predict = Dense(7, activation='softmax')(block3)
    model = Model(input=input_img, output=predict)

    '''
    # better version
    block1 = Conv2D(64, 5, 5, border_mode='valid')(input_img)
    block1 = PReLU(init='zero', weights=None)(block1)
    block1 = ZeroPadding2D(padding=(2, 2), dim_ordering='tf')(block1)
    block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
    block1 = ZeroPadding2D(padding=(1, 1), dim_ordering='tf')(block1)

    block2 = Conv2D(64, 3, 3)(block1)
    block2 = PReLU(init='zero', weights=None)(block2)
    block2 = ZeroPadding2D(padding=(1, 1), dim_ordering='tf')(block2)

    block3 = Conv2D(64, 3, 3)(block2)
    block3 = PReLU(init='zero', weights=None)(block3)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
    block3 = ZeroPadding2D(padding=(1, 1), dim_ordering='tf')(block3)

    block4 = Conv2D(128, 3, 3)(block3)
    block4 = PReLU(init='zero', weights=None)(block4)
    block4 = ZeroPadding2D(padding=(1, 1), dim_ordering='tf')(block4)

    block5 = Conv2D(128, 3, 3)(block4)
    block5 = PReLU(init='zero', weights=None)(block5)
    block5 = ZeroPadding2D(padding=(1, 1), dim_ordering='tf')(block5)
    block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
    block5 = Flatten()(block5)

    fc1 = Dense(1024)(block5)
    fc1 = PReLU(init='zero', weights=None)(fc1)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(1024)(fc1)
    fc2 = PReLU(init='zero', weights=None)(fc2)
    fc2 = Dropout(0.5)(fc2)

    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(input=input_img, output=predict)

    '''
    ### VGG-16
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=(48,48,1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1), dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1), dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1), dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1), dim_ordering='tf'))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    '''
    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=0.1)
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model
