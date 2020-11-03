import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from losses import triple_loss, euclidean_dist
from generate_dataset import show_image
import numpy as np
import random
import pickle
from matplotlib import pyplot as plt
from models import baseline_model, resnet_transfer
import time

K.set_image_data_format = "channels_last"


optimiser = Adam(0.00001)


def loss(model, x, y, training):
    y_pred = model(x)

    return triple_loss(y_true=None, y_pred=y_pred)
    # return tf.keras.losses.binary_crossentropy(y_true=y, y_pred=y_pred)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)

    return loss_value, tape.gradient(
        tf.convert_to_tensor(loss_value), model.trainable_variables
    )


def train(siamese_net, epochs=1):

    train_loss_results = []
    train_accuracy_results = []

    for epoch in range(epochs):

        start_time = time.time()
        idx = random.randint(0, 3)

        with open("triplet-{}.pkl".format(idx), "rb") as f:
            train_data = pickle.load(f)

        anchor = train_data[:, 0, :, :]
        positive = train_data[:, 1, :, :]
        negative = train_data[:, 2, :, :]

        # Optimize the model
        loss_value, grads = grad(siamese_net, [anchor, positive, negative], targets=0)
        optimiser.apply_gradients(zip(grads, siamese_net.trainable_variables))

        # if not random.randint(0, 1):
        #     loss_value, grads = grad(siamese_net, [anchor, negative], targets=0)
        #     optimiser.apply_gradients(zip(grads, siamese_net.trainable_variables))

        # else:
        #     loss_value, grads = grad(siamese_net, [anchor, positive], targets=1)
        #     optimiser.apply_gradients(zip(grads, siamese_net.trainable_variables))

        train_loss_results.append(K.sum(loss_value))

        if epoch % 10 == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            print("--- 1 epoch took {:.3f} seconds ---".format(elapsed))
            print("Estimated time left - {:.2f}".format(elapsed * (epochs - epoch)))
            print("EPOCH-{}".format(epoch))
            print("Loss after {} epochs is : {}".format(epoch, K.sum(loss_value)))

    return train_loss_results


def pipeline1():

    try:
        siamese_net = tf.keras.models.load_model("resnet_transfer.h5")
    except:
        siamese_net, net = resnet_transfer()

    train_loss = train(siamese_net, 30)

    siamese_net.save("fine_tuned_networks/resnet_transfer.h5")
    plt.plot(train_loss)
    plt.ylabel("LOSS")
    plt.xlabel("EPOCHS")
    plt.show()


def visualise_outputs():
    siamese_net = tf.keras.models.load_model("fine_tuned_networks/resnet_transfer.h5")
    try:
        net = siamese_net.get_layer("sequential")
    except ValueError:
        net = siamese_net.get_layer("sequential_1")
    # net = siamese_net.get_layer("sequential")

    with open("triplet-{}.pkl".format(2), "rb") as f:
        train_data = pickle.load(f)

    anchor = train_data[:, 0, :, :]
    positive = train_data[:, 1, :, :]
    negative = train_data[:, 2, :, :]

    preds1 = net(anchor)

    preds2 = net(positive)

    preds3 = net(negative)

    for i, img in enumerate(anchor):
        show_image(img)
        show_image(positive[i])
        show_image(negative[i])

        print(
            "Distance between anchor and +ve = {}".format(
                euclidean_dist(preds1[i], preds2[i])
            )
        )
        print(
            "Distance between anchor and -ve = {}".format(
                euclidean_dist(preds1[i], preds3[i])
            )
        )


# pipeline1()
visualise_outputs()
# siamese_net = tf.keras.models.load_model("resnet_transfer.h5")
# siamese_net.summary()