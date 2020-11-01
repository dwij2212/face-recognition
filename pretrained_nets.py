import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


def inception_resnet_v2(path=None):
    # Helper function to download and save weights locally since weights are large in size
    model = tf.keras.applications.InceptionResNetV2(
        include_top=False, weights="imagenet", input_shape=(100, 80, 3)
    )

    # freeze all first few layers to fine-tune
    for l in model.layers[:-10]:
        l.trainable = False

    print("saving model")
    model.save(os.path.join("trained_networks", path))
    model.summary()


# base_model = tf.keras.models.load_model(
#     os.path.join("trained_networks", "inception_resnetv2.h5"), compile=False
# )
# base_model.summary()
# inception_resnet_v2("inception_resnetv2.h5")