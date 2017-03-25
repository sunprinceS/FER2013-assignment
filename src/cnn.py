from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adam, Adadelta

def build_model():
    input_img = Input(shape=(48, 48, 1))

    block1 = Convolution2D(64, 5, 5, border_mode='valid')(input_img)
    block1 = PReLU(init='zero', weights=None)(block1)
    block1 = ZeroPadding2D(padding=(2, 2))(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(block1)

    block2 = ZeroPadding2D(padding=(1, 1))(block1)
    block2 = Convolution2D(64, 3, 3)(block2)
    block2 = PReLU(init='zero', weights=None)(block2)
    block2 = ZeroPadding2D(padding=(1, 1))(block2)
    block2 = Convolution2D(64, 3, 3)(block2)
    block2 = PReLU(init='zero', weights=None)(block2)
    block2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block2)

    block3 = ZeroPadding2D(padding=(1, 1))(block2)
    block3 = Convolution2D(128, 3, 3)(block3)
    block3 = PReLU(init='zero', weights=None)(block3)
    block3 = ZeroPadding2D(padding=(1, 1))(block3)
    block3 = Convolution2D(128, 3, 3)(block3)
    block3 = PReLU(init='zero', weights=None)(block3)

    block4 = ZeroPadding2D(padding=(1, 1))(block3)
    block4 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block4)
    block4 = Flatten()(block4)

    fc1 = Dense(1024)(block4)
    fc1 = PReLU(init='zero', weights=None)(fc1)
    fc1 = Dropout(0.2)(fc1)

    fc2 = Dense(1024)(fc1)
    fc2 = PReLU(init='zero', weights=None)(fc2)
    fc2 = Dropout(0.2)(fc2)

    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)

    model = Model(input=input_img, output=predict)
    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=0.001)
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model
