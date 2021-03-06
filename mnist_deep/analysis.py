import numpy as np
import tensorflow as tf
from mnist_deep.utils import tfw_print


# Favor the use of this method
def get_accuracy(model, dataset):
    accuracy = tf.metrics.Accuracy()
    for x, y in dataset:
        labels, pred = get_pred_and_labels(model, x, y)
        accuracy(pred, labels)
    # Using print would be better, tf.print not necessary here
    tfw_print("Accuracy: {:.2%}".format(accuracy.result()))


@tf.function
def get_pred_and_labels(model, x, y):
    prob = model(x, training=False)
    pred = tf.argmax(prob, axis=1, output_type=tf.int32)
    labels = tf.argmax(y, axis=1, output_type=tf.int32)
    return labels, pred


def get_pred_and_labels2(dataset, model):
    images, labels = tuple(zip(*dataset))
    images = np.concatenate([i for i in images], axis=0)
    labels = np.concatenate([i for i in labels], axis=0)
    nb_test = images.shape[0]
    nb_classes = labels.shape[1]
    labels = np.argmax(labels, axis=1)
    pred = model.predict(images)
    return pred, labels, nb_classes, nb_test


def get_perf_rank(model, dataset, rank):
    pred, labels, nb_classes, nb_test = get_pred_and_labels2(dataset, model)

    correct = []
    for i in range(nb_test):
        idx_pred = np.argsort(pred[i])[nb_classes - rank:]
        correct.append(labels[i] in idx_pred)
    perf = sum(correct) / nb_test
    print("Performance Rank {}: {:.2%}".format(rank, perf))


def main():
    pass


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()
