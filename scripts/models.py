from tensorflow import keras
from tensorflow.keras.layers import (
    Conv2D,
    Input,
    MaxPooling2D,
    Flatten,
    Dense,
    Lambda,
    Layer,
    BatchNormalization,
)
import tensorflow as tf
import os


def resnet_transfer():
    """
    Similar architecture as the base model. Just the CNN which we get embeddings from is resnet instead of a small CNN
    """
    input1 = Input(shape=(100, 80, 3), name="input1")
    input2 = Input(shape=(100, 80, 3), name="input2")
    input3 = Input(shape=(100, 80, 3), name="input3")

    base_model = tf.keras.models.load_model(
        os.path.join("pre_trained_networks", "inception_resnetv2.h5")
    )

    # freeze batchnorm statistics
    for layer in base_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    model = keras.Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(128, activation="tanh"))

    output1 = model(input1)
    output2 = model(input2)
    output3 = model(input3)

    base_siamese_net = keras.Model(
        inputs=[input1, input2, input3], outputs=[output1, output2, output3]
    )

    return base_siamese_net, model


m1, m2 = resnet_transfer()
m2.summary()


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

    base_siamese_net = keras.Model(
        inputs=[input1, input2, input3], outputs=[output1, output2, output3]
    )

    return base_siamese_net, model