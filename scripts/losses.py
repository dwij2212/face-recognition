import numpy as np
from tensorflow.keras import backend as K

ALPHA = 0.2


def euclidean_dist(embedding1, embedding2):
    """
    Returns euclidean distance between 2 N-dimensional embe
    """
    return K.sum(K.square(embedding1 - embedding2))


def triple_loss(y_true, y_pred, alpha=ALPHA):
    """
    Triplet loss function for tensorflow model.
    The y_pred must be of the form [anchor, true, false] to get correct results.
    This returns (anchor-true)^2 + (anchor-false)^2 + alpha
    """

    # assert len(y_pred) == 3, "Shape should be 3,128"

    anchor = y_pred[0]
    true = y_pred[1]
    false = y_pred[2]

    # Finds euclidean distance between both embeddings
    positive_dist = euclidean_dist(anchor, true)
    negative_dist = euclidean_dist(anchor, false)

    # returns final triplet loss function
    return K.max([positive_dist + negative_dist + alpha, 0])


def normal_function(y_true, y_pred):
    """
    Finds Normal Density function f(z) where z = ytrue - ypred
    f(z) = (e ^(-z^2/2) / sqrt(2 pi))
    Finds similarity in 2 embeddings
    """

    z = np.array(y_true) - np.array(y_pred)

    ex = np.exp(-np.sum(np.square(z)) / 2)
    loss = ex / np.sqrt(2 * np.pi)

    return loss


def main():
    t1 = [1, 2, 3]
    t2 = [2, 2, 3]

    # test normal function [f(z = 1) = 0.24197]
    print(normal_function(t1, t2))

    t3 = [0, 3, 3]

    # test triple loss [triple(t1,t2,t3) = 3.1]
    print(triple_loss(None, [t1, t2, t3]))


if __name__ == "__main__":
    main()