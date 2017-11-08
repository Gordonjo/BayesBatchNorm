from sklearn.datasets import make_moons
import numpy as np
import pdb

def download(n_train=500, n_test=500):
    train_x, train_t = make_moons(n_samples=n_train, shuffle=True, noise=0.2, random_state=1234)
    test_x, test_t = make_moons(n_samples=n_test, shuffle=True, noise=0.2, random_state=1234)

    train_set = (train_x.astype('float32'), train_t.astype('int32'))
    test_set = (test_x.astype('float32'), test_t.astype('int32'))

    # pad targets to be 1-hot
    train_set = pad_targets(train_set)
    test_set = pad_targets(test_set)
    return train_set, test_set

def pad_targets(xy):
    """
    Pad the targets to be 1hot.
    :param xy: A tuple containing the x and y matrices.
    :return: The 1hot coded dataset.
    """
    x, y = xy
    classes = np.max(y) + 1
    tmp_data_y = np.zeros((x.shape[0], classes))
    for i, dp in zip(range(len(y)), y):
        r = np.zeros(classes)
        r[dp] = 1
        tmp_data_y[i] = r
    y = tmp_data_y
    return x, y

