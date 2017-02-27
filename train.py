from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD #, Adam

def build_model():
    input_img = Input(shape=(48, 48, 1))

    layer1 = Convolution2D(32, 3, 3, activation='relu')(input_img)
    layer1 = Convolution2D(32, 3, 3, activation='relu')(layer1)
    layer1 = MaxPooling2D((2, 2))(layer1)
    layer1 = Dropout(0.25)(layer1)

    layer2 = Convolution2D(64, 3, 3, activation='relu')(layer1)
    layer2 = Convolution2D(64, 3, 3, activation='relu')(layer2)
    layer2 = MaxPooling2D((2, 2))(layer2)
    layer2 = Dropout(0.25)(layer2)

    layer3 = Flatten()(layer2)
    layer3 = Dense(256, activation='relu')(layer3)
    layer3 = Dropout(0.5)(layer3)
    predict = Dense(7, activation='softmax')(layer3)

    model = Model(input=input_img, output=predict)
    return model

def compile_model(model):
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # adam = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

def train_on_batch(model, X_batch, Y_batch, batch_size):
    model.train_on_batch(X_batch, Y_batch)



# model = Sequential()
# # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# # this applies 32 convolution filters of size 3x3 each.
# model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 100, 100)))
# model.add(Activation('relu'))
# model.add(Convolution2D(32, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Convolution2D(64, 3, 3, border_mode='valid'))
# model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# # Note: Keras does automatic shape inference.
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

# model.add(Dense(10))
# model.add(Activation('softmax'))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd)

# model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)
