from face_detect import FaceDetector
import os
import cv2.cv2 as cv2
import numpy as np
import pickle
import random

# to disable warnings from TF
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

SHAPE = (512, 512)


def read_images(path=None):
    """
    Takes directory as an arguement and gets faces from images in it.
    """
    faces = []
    for file in os.listdir(path):
        if ".JPG" in file:
            # reads all jpg images and finds faces from them. Asks for names of each person
            # to train and predict
            fname = os.path.join(path, file)
            detector = FaceDetector(fname, SHAPE)

            detector.get_cropped()
            outputs = detector.cropped

            for crop in outputs:
                faces.append(crop)

    return faces


def show_image(img=None):
    """
    Takes an image as an arguement and shows the image
    """
    print("Displaying image...")
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pre_process_faces(img):
    """
    Randomly applies preprocessing on image passed as arguement.
    """
    img = tf.image.random_flip_left_right(img)
    # img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_saturation(img, 0, 2)
    return img


def generate_triplets(faces=None, batch_size=None):
    """
    Make triplets such that all of these are nearly similar but 2 are different (negative)
    and 2 are same (positive). Take one of the positives as anchor for triplet loss.
    """
    if faces is None:
        raise NotImplementedError("Faces do not exist")

    # finds number of faces
    size = faces.shape[0]

    triplets = []

    # randomly generates triplets from identified faces. Raises error if there are only 2 faces
    # in training set
    for _ in range(batch_size):
        try:
            indices = random.sample(range(0, size - 1), 2)
        except ValueError as e:
            print("More than 2 faces are required")
            exit()

        anchor = faces[indices[0]]
        negative = faces[indices[1]]

        # pre-process faces
        positive = pre_process_faces(anchor)
        anchor = pre_process_faces(anchor)
        negative = pre_process_faces(negative)

        triplets.append([anchor, positive, negative])

    return np.array(triplets)


def generate_pairs(faces=None, batch_size=8):
    """
    Generates pairs similar to triplet generation
    """
    if faces is None:
        raise NotImplementedError("Faces do not exist")

    # finds number of faces
    size = faces.shape[0]

    pairs = []
    for _ in range(batch_size):
        try:
            indices = random.sample(range(0, size - 1), 2)
        except ValueError as e:
            print("More than 2 faces are required")
            exit()

        anchor = faces[indices[0]]
        negative = faces[indices[1]]

        pairs.append([anchor, negative])

    return np.array(pairs)


# triplets = generate_triplets(faces, 2)


def main():
    # Main function to read images and store it in lists to use later while training.
    # Generating triplet pairs takes lot of time.
    faces = np.array(read_images("."))

    for i in range(5):
        print("Generating triplets number {}.".format(i))
        triplets = generate_triplets(faces, 8)
        with open("triplet-{}.pkl".format(i), "wb") as f:
            pickle.dump(triplets, f)


if __name__ == "__main__":
    main()