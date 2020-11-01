from tensorflow import keras
from tensorflow.keras.layers import (
    Conv2D,
    Input,
    MaxPooling2D,
    Flatten,
    Dense,
    Lambda,
    Layer,
)
import tensorflow as tf
import os


def resnet_transfer():
    input1 = Input(shape=(100, 80, 3), name="input1")
    input2 = Input(shape=(100, 80, 3), name="input2")
    input3 = Input(shape=(100, 80, 3), name="input3")

    base_model = tf.keras.models.load_model(
        os.path.join("trained_networks", "inception_resnetv2.h5")
    )

    model = keras.Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(128, activation="sigmoid"))

    output1 = model(input1)
    output2 = model(input2)
    output3 = model(input3)

    base_siamese_net = keras.Model(
        inputs=[input1, input2, input3], outputs=[output1, output2, output3]
    )

    return base_siamese_net, model


# m1, m2 = resnet_transfer()
# m2.summary()


def baseline_model():

    input1 = Input(shape=(100, 80, 3), name="input1")
    input2 = Input(shape=(100, 80, 3), name="input2")
    input3 = Input(shape=(100, 80, 3), name="input3")

    model = keras.Sequential()
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            input_shape=(100, 80, 3),
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform"))
    model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(100, activation="softmax"))

    output1 = model(input1)
    output2 = model(input2)
    output3 = model(input3)

    # dist = Lambda(lambda tensors: K.square(tensors[0] - tensors[1]))([output1, output2])
    # output = Dense(1, activation="sigmoid")(dist)

    base_siamese_net = keras.Model(
        inputs=[input1, input2, input3], outputs=[output1, output2, output3]
    )

    return base_siamese_net, model