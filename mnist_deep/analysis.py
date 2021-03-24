import numpy as np
import tensorflow as tf


def get_pred_and_labels(dataset, model):
    images, labels = tuple(zip(*dataset))
    images = np.concatenate([i for i in images], axis=0)
    labels = np.concatenate([i for i in labels], axis=0)
    nb_test = images.shape[0]
    nb_classes = labels.shape[1]
    labels = np.argmax(labels, axis=1)
    pred = model.predict(images)
    return pred, labels, nb_classes, nb_test


def get_perf_rank(model, dataset, rank):

    pred, labels, nb_classes, nb_test = get_pred_and_labels(dataset, model)

    correct = []
    for i in range(nb_test):
        idx_pred = np.argsort(pred[i])[nb_classes-rank:]
        correct.append(labels[i] in idx_pred)
    perf = sum(correct) / nb_test
    print("Performance Rank {}: {:.2%}".format(rank, perf))


# Favor the use of this method
def get_accuracy(model, dataset):
    accuracy = tf.metrics.Accuracy()
    for x, y in dataset:
        prob = model(x, training=False)
        pred = tf.argmax(prob, axis=1, output_type=tf.int32)
        labels = tf.argmax(y, axis=1, output_type=tf.int32)
        accuracy(pred, labels)
    print("Accuracy: {:.2%}".format(accuracy.result()))


def main():
    pass


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()