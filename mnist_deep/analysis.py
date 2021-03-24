import numpy as np


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


# TODO Remake the performance computer using the following code
# If well done, we could have only one batch at a time in the memory

# test_accuracy = tf.keras.metrics.Accuracy()
#
# for (x, y) in test_dataset:
#   # training=False is needed only if there are layers with different
#   # behavior during training versus inference (e.g. Dropout).
#   logits = model(x, training=False)
#   prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
#   test_accuracy(prediction, y)
#
# print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


def main():
    pass


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main()