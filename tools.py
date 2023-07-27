import pickle

import lap
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import pickle


def sava_data(filename, data):
    print("Begin to save data：", filename)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()


def load_data(filename):
    print("Begin to load data：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def get_accuracy(labels, prediction):
    cm = confusion_matrix(labels, prediction)

    def linear_assignment(cost_matrix):
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])

    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)

    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    accuracy = np.trace(cm2) / np.sum(cm2)
    return accuracy


def get_MCM_score(labels, predictions):
    accuracy = get_accuracy(labels, predictions)
    MCM = multilabel_confusion_matrix(labels, predictions)
    tn = MCM[:, 0, 0]
    fp = MCM[:, 0, 1]
    fn = MCM[:, 1, 0]
    tp = MCM[:, 1, 1]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = recall * precision * 2 / (recall + precision)

    return {
        "M_fpr": format(recall * 100, '.3f'),
        "M_fnr": format(precision * 100, '.3f'),
        "M_f1": format(f1 * 100, '.3f'),
        "ACC": format(accuracy * 100, '.3f'),
        "MCM": MCM
    }
